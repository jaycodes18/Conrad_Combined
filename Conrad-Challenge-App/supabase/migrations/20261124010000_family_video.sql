-- Live camera for family. The teen phone broadcasts low-resolution JPEG frames on the private Realtime
-- topic `video:<driver uuid>`. Nothing is stored: Broadcast messages are relayed, not written to a table.
-- Two consents gate it: the driver turns on `video_sharing` for a link (here), and turns the camera on for
-- each drive in the app. Realtime Authorization checks the RLS policies below when a client joins the
-- private channel: only the driver may send on their own topic, and only guardians with an accepted link
-- that has video sharing on may receive.

alter table public.guardian_links add column if not exists video_sharing boolean not null default false;

create or replace function public.guardian_set_video_sharing(p_link_id uuid, p_enabled boolean)
returns void
language plpgsql
security definer
set search_path = ''
as $$
begin
  update public.guardian_links
     set video_sharing = p_enabled
   where id = p_link_id and status = 'accepted' and driver_id = auth.uid();
  if not found then raise exception 'active driver link unavailable' using errcode = 'P0002'; end if;
end;
$$;

-- Revoking a link also ends video access.
create or replace function public.guardian_links_clear_video()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  if new.status <> 'accepted' then new.video_sharing := false; end if;
  return new;
end;
$$;
drop trigger if exists guardian_links_clear_video on public.guardian_links;
create trigger guardian_links_clear_video before update on public.guardian_links
  for each row execute function public.guardian_links_clear_video();

drop function if exists public.guardian_list_links();
create function public.guardian_list_links()
returns table (
  id uuid, driver_id uuid, guardian_id uuid, status text,
  location_sharing boolean, created_at timestamptz, accepted_at timestamptz,
  other_email text, video_sharing boolean
)
language sql
stable
security definer
set search_path = ''
as $$
  select l.id, l.driver_id, l.guardian_id, l.status, l.location_sharing, l.created_at, l.accepted_at,
         case when l.driver_id = u.id then l.guardian_email else d.email end, l.video_sharing
    from public.guardian_links l
    join auth.users u on u.id = auth.uid()
    join auth.users d on d.id = l.driver_id
   where u.email_confirmed_at is not null
     and (l.driver_id = u.id or l.guardian_id = u.id or
          (l.status = 'pending' and l.expires_at > now() and l.guardian_email = lower(u.email)))
   order by l.created_at desc
$$;

drop function if exists public.guardian_list_driver_status();
create function public.guardian_list_driver_status()
returns table (
  driver_id uuid, driver_email text, location_sharing boolean, session_id uuid, active boolean,
  condition text, tier smallint, score real, indicators jsonb, speed_kmh real,
  location_lat double precision, location_lng double precision, heading real, updated_at timestamptz,
  video_sharing boolean
)
language sql
stable
security definer
set search_path = ''
as $$
  select l.driver_id, d.email, l.location_sharing, s.session_id, s.active,
         s.condition, s.tier, s.score, s.indicators, s.speed_kmh,
         case when l.location_sharing then s.location_lat end,
         case when l.location_sharing then s.location_lng end,
         case when l.location_sharing then s.heading end,
         s.updated_at, l.video_sharing
    from public.guardian_links l
    join auth.users d on d.id = l.driver_id
    left join public.driver_live_status s on s.driver_id = l.driver_id
   where l.guardian_id = auth.uid() and l.status = 'accepted'
   order by d.email
$$;

revoke all on function public.guardian_set_video_sharing(uuid, boolean) from public, anon;
revoke all on function public.guardian_list_links() from public, anon;
revoke all on function public.guardian_list_driver_status() from public, anon;
grant execute on function public.guardian_set_video_sharing(uuid, boolean) to authenticated;
grant execute on function public.guardian_list_links() to authenticated;
grant execute on function public.guardian_list_driver_status() to authenticated;

-- Helper used by the Realtime policies. SECURITY DEFINER so the policy can read guardian_links, which
-- clients cannot select from directly for other users' rows.
create or replace function public.can_receive_driver_video(p_topic text)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1 from public.guardian_links l
     where l.status = 'accepted' and l.video_sharing and l.guardian_id = auth.uid()
       and p_topic = 'video:' || l.driver_id::text
  )
$$;
revoke all on function public.can_receive_driver_video(text) from public, anon;
grant execute on function public.can_receive_driver_video(text) to authenticated;

do $$
begin
  if to_regclass('realtime.messages') is not null then
    execute 'drop policy if exists driven_video_send on realtime.messages';
    execute $p$
      create policy driven_video_send on realtime.messages
        for insert to authenticated
        with check (
          realtime.messages.extension = 'broadcast'
          and (select realtime.topic()) = 'video:' || (select auth.uid())::text
        )
    $p$;
    execute 'drop policy if exists driven_video_receive on realtime.messages';
    execute $p$
      create policy driven_video_receive on realtime.messages
        for select to authenticated
        using (
          realtime.messages.extension = 'broadcast'
          and (
            (select realtime.topic()) = 'video:' || (select auth.uid())::text
            or public.can_receive_driver_video((select realtime.topic()))
          )
        )
    $p$;
  end if;
end;
$$;
