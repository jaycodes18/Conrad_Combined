import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { PGlite } from '@electric-sql/pglite';

const migration = await readFile(resolve('supabase/migrations/20261107010000_guardian_test_backend.sql'), 'utf8');

async function database() {
  const db = new PGlite();
  await db.exec(`
    create role anon nologin;
    create role authenticated nologin;
    create publication supabase_realtime;
    create schema auth;
    create table auth.users (
      id uuid primary key, email text not null, email_confirmed_at timestamptz
    );
    create function auth.uid() returns uuid language sql stable as $$
      select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid
    $$;
    grant usage on schema auth to authenticated;
    grant execute on function auth.uid() to authenticated;
  `);
  await db.exec(migration);
  await db.exec(`
    insert into auth.users values
      ('00000000-0000-0000-0000-000000000001', 'driver@example.test', now()),
      ('00000000-0000-0000-0000-000000000002', 'guardian@example.test', now()),
      ('00000000-0000-0000-0000-000000000003', 'intruder@example.test', now()),
      ('00000000-0000-0000-0000-000000000004', 'unverified@example.test', null),
      ('00000000-0000-0000-0000-000000000005', 'secondguardian@example.test', now());
  `);
  return db;
}

async function asUser(db, userId, sql) {
  await db.exec(`set role authenticated; select set_config('request.jwt.claim.sub', '${userId}', false);`);
  try { return await db.query(sql); }
  finally { await db.exec('reset role; reset request.jwt.claim.sub;'); }
}

test('verified driver invites by email; only matching verified guardian accepts', async () => {
  const db = await database();
  try {
    const publication = await db.query("select tablename from pg_publication_tables where pubname = 'supabase_realtime'");
    assert.deepEqual(publication.rows, [{ tablename: 'guardian_test_alerts' }]);
    const invite = await asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_invite(' Guardian@Example.Test ') as id`);
    const linkId = invite.rows[0].id;
    const visible = await asUser(db, '00000000-0000-0000-0000-000000000002',
      'select id, status, other_email from public.guardian_list_links()');
    assert.deepEqual(visible.rows, [{ id: linkId, status: 'pending', other_email: 'driver@example.test' }]);
    await assert.rejects(asUser(db, '00000000-0000-0000-0000-000000000003',
      `select public.guardian_accept('${linkId}')`), /invite unavailable/);
    await asUser(db, '00000000-0000-0000-0000-000000000002',
      `select public.guardian_accept('${linkId}')`);
    const accepted = await db.query('select status, guardian_id from public.guardian_links where id = $1', [linkId]);
    assert.equal(accepted.rows[0].status, 'accepted');
    assert.equal(accepted.rows[0].guardian_id, '00000000-0000-0000-0000-000000000002');
  } finally { await db.close(); }
});

test('unverified accounts cannot invite; an accepted invite cannot be accepted twice', async () => {
  const db = await database();
  try {
    await assert.rejects(asUser(db, '00000000-0000-0000-0000-000000000004',
      `select public.guardian_invite('guardian@example.test')`), /verified email required/);
    const { rows } = await asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_invite('guardian@example.test') as id`);
    const id = rows[0].id;
    await asUser(db, '00000000-0000-0000-0000-000000000002', `select public.guardian_accept('${id}')`);
    await assert.rejects(asUser(db, '00000000-0000-0000-0000-000000000002',
      `select public.guardian_accept('${id}')`), /invite unavailable/);
  } finally { await db.close(); }
});

test('either verified participant can revoke a pending invite without disclosing it to outsiders', async () => {
  const db = await database();
  try {
    const { rows } = await asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_invite('guardian@example.test') as id`);
    const id = rows[0].id;
    await assert.rejects(asUser(db, '00000000-0000-0000-0000-000000000003',
      `select public.guardian_revoke('${id}')`), /link unavailable/);
    await asUser(db, '00000000-0000-0000-0000-000000000002', `select public.guardian_revoke('${id}')`);
    const { rows: links } = await asUser(db, '00000000-0000-0000-0000-000000000001',
      'select status from public.guardian_list_links()');
    assert.deepEqual(links, [{ status: 'revoked' }]);
  } finally { await db.close(); }
});

test('expired invites cannot be listed or accepted; driver can clear the expired row', async () => {
  const db = await database();
  try {
    const { rows } = await asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_invite('guardian@example.test') as id`);
    const id = rows[0].id;
    await db.query('update public.guardian_links set expires_at = now() - interval \'1 second\' where id = $1', [id]);
    const hidden = await asUser(db, '00000000-0000-0000-0000-000000000002',
      'select * from public.guardian_list_links()');
    assert.equal(hidden.rows.length, 0);
    await assert.rejects(asUser(db, '00000000-0000-0000-0000-000000000002',
      `select public.guardian_accept('${id}')`), /invite unavailable/);
    await asUser(db, '00000000-0000-0000-0000-000000000001', `select public.guardian_revoke('${id}')`);
  } finally { await db.close(); }
});

test('RLS isolates links and inbox; alerts require an accepted link and opt-in location', async () => {
  const db = await database();
  try {
    const { rows } = await asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_invite('guardian@example.test') as id`);
    const id = rows[0].id;
    await asUser(db, '00000000-0000-0000-0000-000000000002', `select public.guardian_accept('${id}')`);
    const sessionId = '10000000-0000-0000-0000-000000000001';
    await assert.rejects(asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_send_test_alert('${id}'::uuid, '${sessionId}'::uuid, 'eye-closure'::text, 2::smallint, 8000::integer, 32.8::double precision, -96.8::double precision)`), /location sharing is not enabled/);
    await asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_set_location_sharing('${id}', true)`);
    await assert.rejects(asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_send_test_alert('${id}'::uuid, '${sessionId}'::uuid, 'eye-closure'::text, 2::smallint, 8000::integer, 32.8::double precision, null)`), /guardian_alert_location_pair/);
    await asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_send_test_alert('${id}'::uuid, '${sessionId}'::uuid, 'eye-closure'::text, 2::smallint, 8000::integer, 32.8::double precision, -96.8::double precision)`);
    const secondInvite = await asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_invite('secondguardian@example.test') as id`);
    await asUser(db, '00000000-0000-0000-0000-000000000005',
      `select public.guardian_accept('${secondInvite.rows[0].id}')`);
    await asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_send_test_alert('${secondInvite.rows[0].id}'::uuid, '${sessionId}'::uuid, 'eye-closure'::text, 2::smallint, 8000::integer)`);
    // Same link/session is idempotent and does not create a second row.
    await asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_send_test_alert('${id}'::uuid, '${sessionId}'::uuid, 'head-away'::text, 2::smallint, 9000::integer)`);
    await assert.rejects(asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_send_test_alert('${id}'::uuid, '10000000-0000-0000-0000-000000000002'::uuid, 'eye-closure'::text, 1::smallint, 8000::integer)`), /only L2/);
    const guardianInbox = await asUser(db, '00000000-0000-0000-0000-000000000002',
      'select count(*)::int as count, bool_and(is_test) as test, max(location_lat) as latitude from public.guardian_test_alerts');
    assert.deepEqual(guardianInbox.rows[0], { count: 1, test: true, latitude: 32.8 });
    const outsiderInbox = await asUser(db, '00000000-0000-0000-0000-000000000003',
      'select count(*)::int as count from public.guardian_test_alerts');
    assert.equal(outsiderInbox.rows[0].count, 0);
    const outsiderLinks = await asUser(db, '00000000-0000-0000-0000-000000000003',
      'select count(*)::int as count from public.guardian_links');
    assert.equal(outsiderLinks.rows[0].count, 0);
    await asUser(db, '00000000-0000-0000-0000-000000000002', `select public.guardian_revoke('${id}')`);
    await assert.rejects(asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_send_test_alert('${id}'::uuid, '10000000-0000-0000-0000-000000000002'::uuid, 'eye-closure'::text, 2::smallint, 8000::integer)`), /active accepted link unavailable/);
    const hiddenAfterRevoke = await asUser(db, '00000000-0000-0000-0000-000000000002',
      'select count(*)::int as count from public.guardian_test_alerts');
    assert.equal(hiddenAfterRevoke.rows[0].count, 0);
  } finally { await db.close(); }
});

test('client roles cannot write tables directly and alert metadata is constrained', async () => {
  const db = await database();
  try {
    await assert.rejects(asUser(db, '00000000-0000-0000-0000-000000000001',
      `insert into public.guardian_links(driver_id, guardian_email) values ('00000000-0000-0000-0000-000000000001', 'x@example.test')`), /permission denied/);
    const { rows } = await asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_invite('guardian@example.test') as id`);
    await asUser(db, '00000000-0000-0000-0000-000000000002', `select public.guardian_accept('${rows[0].id}')`);
    await assert.rejects(asUser(db, '00000000-0000-0000-0000-000000000001',
      `select public.guardian_send_test_alert('${rows[0].id}'::uuid, '10000000-0000-0000-0000-000000000003'::uuid, 'real-emergency'::text, 2::smallint, 8000::integer)`), /guardian_test_alerts_cue_check/);
  } finally { await db.close(); }
});
