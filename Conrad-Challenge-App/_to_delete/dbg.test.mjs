import { RiskEngine } from './src/risk.ts';
const e = new RiskEngine();
const n = { eyesClosed: 0.08, yaw: 0, pitch: 0 }, c = { eyesClosed: 0.95, yaw: 0, pitch: 0 };
for (const at of [0,700,1400,2000]) console.log(at, e.update({at, face:n}).coverage);
for (let at=2100; at<=7100; at+=100){ const r=e.update({at,face:c}); if(r.alerts.length||at%1000===100) console.log(at, r.coverage, JSON.stringify(r.alerts), r.snapshot.eyesClosedMs); }
