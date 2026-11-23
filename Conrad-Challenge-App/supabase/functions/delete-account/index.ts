import { createClient } from 'npm:@supabase/supabase-js@2.119.0';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, apikey, content-type, x-client-info',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

Deno.serve(async (request: Request) => {
  if (request.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders });
  if (request.method !== 'POST') return Response.json({ error: 'method not allowed' }, { status: 405, headers: corsHeaders });

  const authorization = request.headers.get('Authorization');
  const token = authorization?.match(/^Bearer\s+(.+)$/i)?.[1];
  if (!token) return Response.json({ error: 'authentication required' }, { status: 401, headers: corsHeaders });

  const url = Deno.env.get('SUPABASE_URL');
  const publishableKey = Deno.env.get('SUPABASE_ANON_KEY');
  const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
  if (!url || !publishableKey || !serviceRoleKey) {
    return Response.json({ error: 'server configuration unavailable' }, { status: 500, headers: corsHeaders });
  }

  const callerClient = createClient(url, publishableKey, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
  const { data: { user }, error: authError } = await callerClient.auth.getUser(token);
  if (authError || !user) return Response.json({ error: 'invalid session' }, { status: 401, headers: corsHeaders });

  const adminClient = createClient(url, serviceRoleKey, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
  // Revoke all current refreshable sessions before deleting the verified caller.
  const { error: revokeError } = await adminClient.auth.admin.signOut(token, 'global');
  if (revokeError) return Response.json({ error: 'could not revoke sessions' }, { status: 500, headers: corsHeaders });

  const { error: deleteError } = await adminClient.auth.admin.deleteUser(user.id);
  if (deleteError) return Response.json({ error: 'account deletion failed' }, { status: 500, headers: corsHeaders });
  return Response.json({ deleted: true }, { status: 200, headers: corsHeaders });
});
