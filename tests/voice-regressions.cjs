const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync('live_demo/voice.js', 'utf8');
function setup() {
  const sources = [], sent = [], statuses = [];
  const context = vm.createContext({window: {}, clearTimeout, setTimeout, Uint8Array, DataView,
    atob: s => Buffer.from(s, 'base64').toString('binary')});
  vm.runInContext(source, context);
  const voice = new context.window.ClaimVoice({send: m => sent.push(m), status: s => statuses.push(s), error() {}});
  voice.context = {currentTime: 1, destination: {}, close: async () => {},
    createBuffer: (c, n, rate) => ({duration: n / rate, getChannelData: () => new Float32Array(n)}),
    createBufferSource: () => {const s = {connect() {}, start() {}, stop() {this.stopped = true;}}; sources.push(s); return s;}};
  voice.enabled = voice.ready = true;
  return {voice, sources, sent, statuses};
}
{
  const {voice, sources} = setup();
  voice.begin(2); voice.play({epoch: 2, data: 'AAAAAA=='});
  assert.equal(sources.length, 1);
  voice.interrupt(); assert(sources[0].stopped);
  voice.play({epoch: 2, data: 'AAAAAA=='}); assert.equal(sources.length, 1);
  voice.begin(4); voice.play({epoch: 2, data: 'AAAAAA=='}); assert.equal(sources.length, 1);
  voice.play({epoch: 4, data: 'AAAAAA=='}); assert.equal(sources.length, 2);
  voice.stop(false);
  console.log('PASS interruption stops playback and rejects late audio');
}
{
  const {voice, sent} = setup(); let stopped = 0;
  voice.stream = {getTracks: () => [{stop() {stopped++;}}]};
  voice.stop(); assert.equal(stopped, 1); assert.equal(voice.ready, false);
  assert.equal(sent[0].type, 'voice_state'); assert.equal(sent[0].enabled, false);
  console.log('PASS stop releases microphone and notifies server');
}
{
  const {voice, sources} = setup();
  voice.begin(1); voice.play({epoch: 1, data: 'AA=='}); assert.equal(sources.length, 0);
  voice.stop(false);
  console.log('PASS incomplete PCM sample is rejected');
}
