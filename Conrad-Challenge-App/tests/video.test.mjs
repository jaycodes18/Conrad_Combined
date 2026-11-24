import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { PGlite } from '@electric-sql/pglite';

const migrations = await Promise.all([
  'supabase/migrations/20261107010000_guardian_test_backend.sql',
  'supabase/migrations/20261124000000_family_live_status.sql',
  'supabase/migrations/20261124010000_family_video.sql',
].map((path) => readFile(resolve(path), 'utf8')));

const DRIVER = '00000000-0000-0000-0000-000000000001';
const GUARDIAN = '00000000-0000-0000-0000-000000000002';
const OUTSIDER = '00000000-0000-0000-0000-000000000003';

// A stand-in for Supabase's realtime schema: Realtime Authorization evaluates RLS on realtime.messages with
// realtime.topic() returning the channel topic being joined.
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
    create schema realtime;
    create table realtime.messages (id serial primary key, topic text not null, extension text not null, event text, payload jsonb);
    alter table realtime.messages enable row level security;
    create function realtime.topic() returns text language sql stable as $$ select current_setting('realtime.topic', true) $$;
    grant usage on schema realtime to authenticated;
    grant select, insert on realtime.messages to authenticated;
    grant usage on sequence realtime.messages_id_seq to authenticated;
    grant execute on function realtime.topic() to authenticated;
  `);
  for (const sql of migrations) await db.exec(sql);
  await db.exec(`insert into auth.users values
    ('${DRIVER}', 'maya@example.test', now()), ('${GUARDIAN}', 'parent@example.test', now()),
    ('${OUTSIDER}', 'intruder@example.test', now());`);
  await db.exec(`insert into realtime.messages (topic, extension, event) values ('video:${DRIVER}', 'broadcast', 'frame');`);
  return db;
}

async function asUser(db, userId, topic, sql) {
  await db.exec(`set role authenticated; select set_config('request.jwt.claim.sub', '${userId}', false);
    select set_config('realtime.topic', '${topic}', false);`);
  try { return await db.query(sql); }
  finally { await db.exec('reset role; reset request.jwt.claim.sub; reset realtime.topic;'); }
}

const canJoin = async (db, user, topic) =>
  (await asUser(db, user, topic, `select count(*)::int as n from realtime.messages where topic = '${topic}'`)).rows[0].n > 0;

async function link(db) {
  const { rows } = await asUser(db, DRIVER, '', `select public.guardian_invite('parent@example.test') as id`);
  await asUser(db, GUARDIAN, '', `select public.guardian_accept('${rows[0].id}')`);
  return rows[0].id;
}

test('a guardian receives the driver camera only after the driver turns video sharing on', async () => {
  const db = await database();
  try {
    const id = await link(db);
    const topic = `video:${DRIVER}`;
    assert.equal(await canJoin(db, GUARDIAN, topic), false);
    await asUser(db, DRIVER, '', `select public.guardian_set_video_sharing('${id}', true)`);
    assert.equal(await canJoin(db, GUARDIAN, topic), true);
    assert.equal(await canJoin(db, OUTSIDER, topic), false);
    const listed = await asUser(db, GUARDIAN, '', 'select video_sharing from public.guardian_list_driver_status()');
    assert.deepEqual(listed.rows, [{ video_sharing: true }]);
  } finally { await db.close(); }
});

test('only the driver can send on their own video topic', async () => {
  const db = await database();
  try {
    const id = await link(db);
    await asUser(db, DRIVER, '', `select public.guardian_set_video_sharing('${id}', true)`);
    await asUser(db, DRIVER, `video:${DRIVER}`,
      `insert into realtime.messages (topic, extension, event) values ('video:${DRIVER}', 'broadcast', 'frame')`);
    await assert.rejects(asUser(db, GUARDIAN, `video:${DRIVER}`,
      `insert into realtime.messages (topic, extension, event) values ('video:${DRIVER}', 'broadcast', 'frame')`), /row-level security/);
    await assert.rejects(asUser(db, OUTSIDER, `video:${DRIVER}`,
      `insert into realtime.messages (topic, extension, event) values ('video:${DRIVER}', 'broadcast', 'frame')`), /row-level security/);
  } finally { await db.close(); }
});

test('revoking the link or turning sharing off ends video access; guardians cannot toggle it', async () => {
  const db = await database();
  try {
    const id = await link(db);
    await assert.rejects(asUser(db, GUARDIAN, '', `select public.guardian_set_video_sharing('${id}', true)`), /active driver link unavailable/);
    await asUser(db, DRIVER, '', `select public.guardian_set_video_sharing('${id}', true)`);
    await asUser(db, DRIVER, '', `select public.guardian_set_video_sharing('${id}', false)`);
    assert.equal(await canJoin(db, GUARDIAN, `video:${DRIVER}`), false);
    await asUser(db, DRIVER, '', `select public.guardian_set_video_sharing('${id}', true)`);
    await asUser(db, GUARDIAN, '', `select public.guardian_revoke('${id}')`);
    assert.equal(await canJoin(db, GUARDIAN, `video:${DRIVER}`), false);
    const row = await db.query('select video_sharing from public.guardian_links where id = $1', [id]);
    assert.equal(row.rows[0].video_sharing, false);
  } finally { await db.close(); }
});
