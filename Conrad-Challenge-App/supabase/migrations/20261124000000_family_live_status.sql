-- Family live view: the teen phone publishes the model's current reading during a session that the
-- driver chose to share, and a one-row summary when the session ends. Accepted guardians read both through
-- SECURITY DEFINER functions. No camera pixels, audio or landmarks are stored; `indicators` holds the
-- six-tile text the driver screen shows. Location is stored only when at least one accepted link allows it,
-- and is returned only to guardians whose own link allows it.

create table if not exists public.driver_live_status (
  driver_id uuid primary key references auth.users(id) on delete cascade,
  session_id uuid not null,
  active boolean not null,
  condition text not null check (condition in
    ('none', 'distracted', 'drowsy', 'possible_impairment', 'possible_unresponsiveness')),
  tier smallint not null check (tier between 0 and 3),
  score real not null check (score between 0 and 1),
  indicators jsonb not null default '{}'::jsonb
    check (jsonb_typeof(indicators) = 'object' and pg_column_size(indicators) <= 4096),
  speed_kmh real check (speed_kmh is null or speed_kmh between 0 and 400),
  location_lat double precision,
  location_lng double precision,
  heading real check (heading is null or heading between -360 and 360),
  updated_at timestamptz not null default now(),
  constraint driver_live_location_pair check (
    (location_lat is null and location_lng is null) or
    (location_lat is not null and location_lng is not null and
     location_lat between -90 and 90 and location_lng between -180 and 180)
  )
);

create table if not exists public.driver_trip_summaries (
  id uuid primary key default gen_random_uuid(),
  driver_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid not null,
  started_at timestamptz not null,
  duration_s integer not null check (duration_s between 0 and 86400),
  driver_alerts integer not null check (driver_alerts between 0 and 1000),
  guardian_alerts integer not null check (guardian_alerts between 0 and 1000),
  worst_condition text not null check (worst_condition in
    ('none', 'distracted', 'drowsy', 'possible_impairment', 'possible_unresponsiveness')),
  tracked_pct real check (tracked_pct is null or tracked_pct between 0 and 100),
  created_at timestamptz not null default now(),
  constraint driver_trip_once_per_session unique (driver_id, session_id)
);
create index if not exists driver_trip_summaries_recent_idx
  on public.driver_trip_summaries (driver_id, started_at desc);

alter table public.driver_live_status enable row level security;
alter table public.driver_trip_summaries enable row level security;
revoke all on public.driver_live_status from anon, authenticated;
revoke all on public.driver_trip_summaries from anon, authenticated;

create or replace function public.driver_publish_status(
  p_session_id uuid, p_active boolean, p_condition text, p_tier smallint, p_score real,
  p_indicators jsonb, p_speed_kmh real default null,
  p_location_lat double precision default null, p_location_lng double precision default null,
  p_heading real default null
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
begin
  if auth.uid() is null then raise exception 'authentication required' using errcode = '42501'; end if;
  if not exists (select 1 from public.guardian_links l where l.driver_id = auth.uid() and l.status = 'accepted') then
    raise exception 'no accepted guardian link' using errcode = '42501';
  end if;
  if p_location_lat is not null and not exists (
    select 1 from public.guardian_links l
     where l.driver_id = auth.uid() and l.status = 'accepted' and l.location_sharing
  ) then
    raise exception 'location sharing is not enabled for any link' using errcode = '42501';
  end if;
  insert into public.driver_live_status as s (
    driver_id, session_id, active, condition, tier, score, indicators,
    speed_kmh, location_lat, location_lng, heading, updated_at
  ) values (
    auth.uid(), p_session_id, p_active, p_condition, p_tier, p_score, coalesce(p_indicators, '{}'::jsonb),
    p_speed_kmh, p_location_lat, p_location_lng, p_heading, now()
  )
  on conflict (driver_id) do update set
    session_id = excluded.session_id, active = excluded.active, condition = excluded.condition,
    tier = excluded.tier, score = excluded.score, indicators = excluded.indicators,
    speed_kmh = excluded.speed_kmh, location_lat = excluded.location_lat,
    location_lng = excluded.location_lng, heading = excluded.heading, updated_at = now();
end;
$$;

create or replace function public.driver_record_trip(
  p_session_id uuid, p_started_at timestamptz, p_duration_s integer, p_driver_alerts integer,
  p_guardian_alerts integer, p_worst_condition text, p_tracked_pct real default null
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
begin
  if auth.uid() is null then raise exception 'authentication required' using errcode = '42501'; end if;
  if not exists (select 1 from public.guardian_links l where l.driver_id = auth.uid() and l.status = 'accepted') then
    raise exception 'no accepted guardian link' using errcode = '42501';
  end if;
  insert into public.driver_trip_summaries (
    driver_id, session_id, started_at, duration_s, driver_alerts, guardian_alerts, worst_condition, tracked_pct
  ) values (
    auth.uid(), p_session_id, p_started_at, p_duration_s, p_driver_alerts, p_guardian_alerts,
    p_worst_condition, p_tracked_pct
  ) on conflict (driver_id, session_id) do nothing;
  update public.driver_live_status
     set active = false, speed_kmh = null, updated_at = now()
   where driver_id = auth.uid() and session_id = p_session_id;
end;
$$;

create or replace function public.guardian_list_driver_status()
returns table (
  driver_id uuid, driver_email text, location_sharing boolean, session_id uuid, active boolean,
  condition text, tier smallint, score real, indicators jsonb, speed_kmh real,
  location_lat double precision, location_lng double precision, heading real, updated_at timestamptz
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
         s.updated_at
    from public.guardian_links l
    join auth.users d on d.id = l.driver_id
    left join public.driver_live_status s on s.driver_id = l.driver_id
   where l.guardian_id = auth.uid() and l.status = 'accepted'
   order by d.email
$$;

create or replace function public.guardian_list_driver_trips(p_driver_id uuid)
returns table (
  session_id uuid, started_at timestamptz, duration_s integer, driver_alerts integer,
  guardian_alerts integer, worst_condition text, tracked_pct real
)
language sql
stable
security definer
set search_path = ''
as $$
  select t.session_id, t.started_at, t.duration_s, t.driver_alerts, t.guardian_alerts, t.worst_condition, t.tracked_pct
    from public.driver_trip_summaries t
   where t.driver_id = p_driver_id and exists (
     select 1 from public.guardian_links l
      where l.driver_id = p_driver_id and l.guardian_id = auth.uid() and l.status = 'accepted'
   )
   order by t.started_at desc
   limit 30
$$;

revoke all on function public.driver_publish_status(uuid, boolean, text, smallint, real, jsonb, real, double precision, double precision, real) from public, anon;
revoke all on function public.driver_record_trip(uuid, timestamptz, integer, integer, integer, text, real) from public, anon;
revoke all on function public.guardian_list_driver_status() from public, anon;
revoke all on function public.guardian_list_driver_trips(uuid) from public, anon;
grant execute on function public.driver_publish_status(uuid, boolean, text, smallint, real, jsonb, real, double precision, double precision, real) to authenticated;
grant execute on function public.driver_record_trip(uuid, timestamptz, integer, integer, integer, text, real) to authenticated;
grant execute on function public.guardian_list_driver_status() to authenticated;
grant execute on function public.guardian_list_driver_trips(uuid) to authenticated;
