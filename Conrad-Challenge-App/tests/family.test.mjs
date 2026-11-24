import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { PGlite } from '@electric-sql/pglite';
import { demoFamily, isLive } from '../src/family.ts';

const migrations = await Promise.all([
  'supabase/migrations/20261107010000_guardian_test_backend.sql',
  'supabase/migrations/20261124000000_family_live_status.sql',
].map((path) => readFile(resolve(path), 'utf8')));

const DRIVER = '00000000-0000-0000-0000-000000000001';
const GUARDIAN = '00000000-0000-0000-0000-000000000002';
const OUTSIDER = '00000000-0000-0000-0000-000000000003';
const SECOND = '00000000-0000-0000-0000-000000000005';
const SESSION = '20000000-0000-0000-0000-000000000001';

async function database() {
  const db = new PGlite();
  await db.exec(`
    create role anon nologin;
    create role authenticated nologin;
    create publication supabase_realtime;
    create schema auth;
    create table auth.users (id uuid primary key, email text not null, email_confirmed_at timestamptz);
    create function auth.uid() returns uuid language sql stable as $$
      select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid
    $$;
    grant usage on schema auth to authenticated;
    grant execute on function auth.uid() to authenticated;
  `);
  for (const sql of migrations) await db.exec(sql);
  await db.exec(`
    insert into auth.users values
      ('${DRIVER}', 'maya.r@example.test', now()),
      ('${GUARDIAN}', 'parent@example.test', now()),
      ('${OUTSIDER}', 'intruder@example.test', now()),
      ('${SECOND}', 'aunt@example.test', now());
  `);
  return db;
}

async function asUser(db, userId, sql) {
  await db.exec(`set role authenticated; select set_config('request.jwt.claim.sub', '${userId}', false);`);
  try { return await db.query(sql); }
  finally { await db.exec('reset role; reset request.jwt.claim.sub;'); }
}

async function link(db, guardian, email) {
  const { rows } = await asUser(db, DRIVER, `select public.guardian_invite('${email}') as id`);
  await asUser(db, guardian, `select public.guardian_accept('${rows[0].id}')`);
  return rows[0].id;
}

const publish = (lat = 'null', lng = 'null', condition = 'drowsy') =>
  `select public.driver_publish_status('${SESSION}'::uuid, true, '${condition}', 1::smallint, 0.6::real,
     '{"eyes":{"value":"Closed 2.4 s","tone":"alert"}}'::jsonb, 52::real, ${lat}::double precision, ${lng}::double precision, 90::real)`;

test('a driver cannot publish live status without an accepted guardian', async () => {
  const db = await database();
  try {
    await assert.rejects(asUser(db, DRIVER, publish()), /no accepted guardian link/);
  } finally { await db.close(); }
});

test('guardians see live status; location only on links that allow it; outsiders see nothing', async () => {
  const db = await database();
  try {
    const first = await link(db, GUARDIAN, 'parent@example.test');
    await link(db, SECOND, 'aunt@example.test');
    await assert.rejects(asUser(db, DRIVER, publish('32.95', '-96.99')), /location sharing is not enabled/);
    await asUser(db, DRIVER, `select public.guardian_set_location_sharing('${first}', true)`);
    await asUser(db, DRIVER, publish('32.95', '-96.99'));

    const parent = await asUser(db, GUARDIAN, 'select driver_email, condition, location_lat, indicators from public.guardian_list_driver_status()');
    assert.equal(parent.rows.length, 1);
    assert.equal(parent.rows[0].driver_email, 'maya.r@example.test');
    assert.equal(parent.rows[0].condition, 'drowsy');
    assert.equal(parent.rows[0].location_lat, 32.95);
    assert.equal(parent.rows[0].indicators.eyes.tone, 'alert');

    const aunt = await asUser(db, SECOND, 'select location_lat, condition from public.guardian_list_driver_status()');
    assert.deepEqual(aunt.rows, [{ location_lat: null, condition: 'drowsy' }]);

    const outsider = await asUser(db, OUTSIDER, 'select * from public.guardian_list_driver_status()');
    assert.equal(outsider.rows.length, 0);
    await assert.rejects(asUser(db, OUTSIDER, 'select * from public.driver_live_status'), /permission denied/);
  } finally { await db.close(); }
});

test('ending a session records one trip, marks status inactive, and trips stay private', async () => {
  const db = await database();
  try {
    await link(db, GUARDIAN, 'parent@example.test');
    await asUser(db, DRIVER, publish());
    const record = `select public.driver_record_trip('${SESSION}'::uuid, now() - interval '20 minutes', 1200, 2, 1, 'possible_unresponsiveness', 96.5::real)`;
    await asUser(db, DRIVER, record);
    await asUser(db, DRIVER, record);
    const trips = await asUser(db, GUARDIAN, `select duration_s, worst_condition from public.guardian_list_driver_trips('${DRIVER}')`);
    assert.deepEqual(trips.rows, [{ duration_s: 1200, worst_condition: 'possible_unresponsiveness' }]);
    const status = await asUser(db, GUARDIAN, 'select active from public.guardian_list_driver_status()');
    assert.equal(status.rows[0].active, false);
    const outsider = await asUser(db, OUTSIDER, `select * from public.guardian_list_driver_trips('${DRIVER}')`);
    assert.equal(outsider.rows.length, 0);
    await assert.rejects(asUser(db, DRIVER,
      `select public.driver_record_trip(gen_random_uuid(), now(), 10, 0, 0, 'drunk', null)`), /worst_condition_check/);
  } finally { await db.close(); }
});

test('revoking the link hides the driver from the guardian', async () => {
  const db = await database();
  try {
    const id = await link(db, GUARDIAN, 'parent@example.test');
    await asUser(db, DRIVER, publish());
    await asUser(db, GUARDIAN, `select public.guardian_revoke('${id}')`);
    const rows = await asUser(db, GUARDIAN, 'select * from public.guardian_list_driver_status()');
    assert.equal(rows.rows.length, 0);
  } finally { await db.close(); }
});

test('the demo family is live, located and uses only allowed condition names', () => {
  const family = demoFamily(1_800_000_000_000);
  assert.equal(family.length, 2);
  for (const driver of family) {
    assert.ok(isLive(driver.status, 1_800_000_000_000));
    assert.ok(driver.status.location);
    assert.equal(driver.status.indicators.length, 6);
    assert.match(driver.status.condition, /^(none|distracted|drowsy|possible_impairment|possible_unresponsiveness)$/);
    assert.ok(driver.trips.length > 0);
  }
});
