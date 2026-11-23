-- Optional guardian backend for explicit, clearly labelled test-session alerts.
-- Apply only to a dedicated Supabase project after reviewing auth email settings.
create table if not exists public.guardian_links (
  id uuid primary key default gen_random_uuid(),
  driver_id uuid not null references auth.users(id) on delete cascade,
  guardian_id uuid references auth.users(id) on delete cascade,
  guardian_email text not null,
  status text not null default 'pending' check (status in ('pending', 'accepted', 'revoked')),
  location_sharing boolean not null default false,
  created_at timestamptz not null default now(),
  expires_at timestamptz not null default (now() + interval '24 hours'),
  accepted_at timestamptz,
  revoked_at timestamptz,
  constraint guardian_link_state check (
    (status = 'pending' and guardian_id is null and accepted_at is null and revoked_at is null) or
    (status = 'accepted' and guardian_id is not null and accepted_at is not null and revoked_at is null) or
    (status = 'revoked' and revoked_at is not null)
  )
);
create unique index if not exists guardian_links_pending_email_uq
  on public.guardian_links (driver_id, guardian_email) where status = 'pending';
create index if not exists guardian_links_pending_recipient_idx
  on public.guardian_links (guardian_email, expires_at) where status = 'pending';
create unique index if not exists guardian_links_active_pair_uq
  on public.guardian_links (driver_id, guardian_id) where status = 'accepted';
create index if not exists guardian_links_guardian_status_idx
  on public.guardian_links (guardian_id, status);

create table if not exists public.guardian_test_alerts (
  id uuid primary key default gen_random_uuid(),
  link_id uuid not null references public.guardian_links(id) on delete cascade,
  driver_id uuid not null references auth.users(id) on delete cascade,
  guardian_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid not null,
  cue text not null check (cue in ('eye-closure', 'head-away')),
  level smallint not null check (level = 2),
  evidence_ms integer not null check (evidence_ms between 0 and 600000),
  is_test boolean not null default true check (is_test),
  location_lat double precision,
  location_lng double precision,
  created_at timestamptz not null default now(),
  constraint guardian_alert_location_pair check (
    (location_lat is null and location_lng is null) or
    (location_lat is not null and location_lng is not null and
     location_lat between -90 and 90 and location_lng between -180 and 180)
  ),
  constraint guardian_alert_once_per_session unique (driver_id, session_id)
);
create index if not exists guardian_test_alerts_inbox_idx
  on public.guardian_test_alerts (guardian_id, created_at desc);

alter table public.guardian_links enable row level security;
alter table public.guardian_test_alerts enable row level security;

drop policy if exists guardian_links_read_participating on public.guardian_links;
create policy guardian_links_read_participating on public.guardian_links
  for select to authenticated
  using ((select auth.uid()) = driver_id or (select auth.uid()) = guardian_id);

drop policy if exists guardian_test_alerts_read_guardian on public.guardian_test_alerts;
create policy guardian_test_alerts_read_guardian on public.guardian_test_alerts
  for select to authenticated
  using (
    (select auth.uid()) = guardian_id and exists (
      select 1 from public.guardian_links l
       where l.id = link_id and l.guardian_id = (select auth.uid()) and l.status = 'accepted'
    )
  );

revoke all on public.guardian_links from anon, authenticated;
revoke all on public.guardian_test_alerts from anon, authenticated;
grant select on public.guardian_links, public.guardian_test_alerts to authenticated;

do $$
begin
  if exists (select 1 from pg_catalog.pg_publication where pubname = 'supabase_realtime')
     and not exists (
       select 1 from pg_catalog.pg_publication_tables
        where pubname = 'supabase_realtime' and schemaname = 'public' and tablename = 'guardian_test_alerts'
     ) then
    alter publication supabase_realtime add table public.guardian_test_alerts;
  end if;
end;
$$;

-- These small, narrowly scoped SECURITY DEFINER RPCs perform checks that need
-- access to auth.users or must update rows without granting broad table writes.
create or replace function public.guardian_invite(p_email text)
returns uuid
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_id uuid;
  v_email text := lower(trim(p_email));
begin
  if auth.uid() is null then raise exception 'authentication required' using errcode = '42501'; end if;
  if not exists (select 1 from auth.users u where u.id = auth.uid() and u.email_confirmed_at is not null) then
    raise exception 'verified email required' using errcode = '42501';
  end if;
  if v_email is null or length(v_email) > 254 or v_email !~ '^[^[:space:]@]+@[^[:space:]@]+\.[^[:space:]@]+$' then
    raise exception 'valid email required' using errcode = '22023';
  end if;
  if v_email = lower((select u.email from auth.users u where u.id = auth.uid())) then
    raise exception 'cannot invite yourself' using errcode = '22023';
  end if;
  insert into public.guardian_links(driver_id, guardian_email)
    values (auth.uid(), v_email)
    on conflict (driver_id, guardian_email) where status = 'pending'
    do update set created_at = now(), expires_at = now() + interval '24 hours'
    returning id into v_id;
  return v_id;
end;
$$;

create or replace function public.guardian_accept(p_link_id uuid)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_user auth.users%rowtype;
begin
  select * into v_user from auth.users where id = auth.uid();
  if v_user.id is null or v_user.email_confirmed_at is null then
    raise exception 'verified account required' using errcode = '42501';
  end if;
  update public.guardian_links
     set guardian_id = v_user.id, status = 'accepted', accepted_at = now()
   where id = p_link_id and status = 'pending' and expires_at > now()
     and guardian_email = lower(v_user.email);
  if not found then raise exception 'invite unavailable for this verified email' using errcode = 'P0002'; end if;
end;
$$;

create or replace function public.guardian_list_links()
returns table (
  id uuid, driver_id uuid, guardian_id uuid, status text,
  location_sharing boolean, created_at timestamptz, accepted_at timestamptz,
  other_email text
)
language sql
stable
security definer
set search_path = ''
as $$
  select l.id, l.driver_id, l.guardian_id, l.status, l.location_sharing, l.created_at, l.accepted_at,
         case when l.driver_id = u.id then l.guardian_email else d.email end
    from public.guardian_links l
    join auth.users u on u.id = auth.uid()
    join auth.users d on d.id = l.driver_id
   where u.email_confirmed_at is not null
     and (l.driver_id = u.id or l.guardian_id = u.id or
          (l.status = 'pending' and l.expires_at > now() and l.guardian_email = lower(u.email)))
   order by l.created_at desc
$$;

create or replace function public.guardian_revoke(p_link_id uuid)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_user auth.users%rowtype;
begin
  select * into v_user from auth.users where id = auth.uid();
  update public.guardian_links
     set status = 'revoked', revoked_at = now(), location_sharing = false
   where id = p_link_id and (
     (status = 'accepted' and (driver_id = auth.uid() or guardian_id = auth.uid())) or
     (status = 'pending' and (
       driver_id = auth.uid() or
       (expires_at > now() and v_user.email_confirmed_at is not null and guardian_email = lower(v_user.email))
     ))
   );
  if not found then raise exception 'link unavailable' using errcode = 'P0002'; end if;
end;
$$;

create or replace function public.guardian_set_location_sharing(p_link_id uuid, p_enabled boolean)
returns void
language plpgsql
security definer
set search_path = ''
as $$
begin
  update public.guardian_links
     set location_sharing = p_enabled
   where id = p_link_id and status = 'accepted' and driver_id = auth.uid();
  if not found then raise exception 'active driver link unavailable' using errcode = 'P0002'; end if;
end;
$$;

create or replace function public.guardian_send_test_alert(
  p_link_id uuid, p_session_id uuid, p_cue text, p_level smallint,
  p_evidence_ms integer, p_location_lat double precision default null,
  p_location_lng double precision default null
)
returns uuid
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_link public.guardian_links%rowtype;
  v_id uuid;
begin
  select * into v_link from public.guardian_links
   where id = p_link_id and driver_id = auth.uid() and status = 'accepted'
   for update;
  if v_link.id is null then raise exception 'active accepted link unavailable' using errcode = '42501'; end if;
  if p_level is distinct from 2 then raise exception 'only L2 test alerts can be sent' using errcode = '22023'; end if;
  if p_location_lat is not null and not v_link.location_sharing then
    raise exception 'location sharing is not enabled for this link' using errcode = '42501';
  end if;
  insert into public.guardian_test_alerts(
    link_id, driver_id, guardian_id, session_id, cue, level, evidence_ms,
    location_lat, location_lng
  ) values (
    v_link.id, v_link.driver_id, v_link.guardian_id, p_session_id, p_cue, p_level, p_evidence_ms,
    p_location_lat, p_location_lng
  ) on conflict (driver_id, session_id) do nothing returning id into v_id;
  return v_id;
end;
$$;

revoke all on function public.guardian_invite(text) from public, anon;
revoke all on function public.guardian_accept(uuid) from public, anon;
revoke all on function public.guardian_list_links() from public, anon;
revoke all on function public.guardian_revoke(uuid) from public, anon;
revoke all on function public.guardian_set_location_sharing(uuid, boolean) from public, anon;
revoke all on function public.guardian_send_test_alert(uuid, uuid, text, smallint, integer, double precision, double precision) from public, anon;
grant execute on function public.guardian_invite(text) to authenticated;
grant execute on function public.guardian_accept(uuid) to authenticated;
grant execute on function public.guardian_list_links() to authenticated;
grant execute on function public.guardian_revoke(uuid) to authenticated;
grant execute on function public.guardian_set_location_sharing(uuid, boolean) to authenticated;
grant execute on function public.guardian_send_test_alert(uuid, uuid, text, smallint, integer, double precision, double precision) to authenticated;
