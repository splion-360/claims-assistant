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
{
  const {voice, sources, sent} = setup();
  voice.begin(5); voice.play({epoch: 5, data: 'AAAAAA=='}); voice.end(5);
  assert.equal(sent.length, 0);
  sources[0].onended();
  assert.equal(sent[0].type, 'playback_done'); assert.equal(sent[0].epoch, 5);
  voice.stop(false);
  console.log('PASS completed playback acknowledged only after audio finishes');
}
{
  let Processor;
  const context = vm.createContext({AudioWorkletProcessor: class {constructor() {this.port = {postMessage: x => output.push(x)};}},
    sampleRate: 48000, registerProcessor: (name, cls) => Processor = cls, Int16Array});
  const output = [];
  vm.runInContext(fs.readFileSync('live_demo/audio-worklet.js', 'utf8'), context);
  const processor = new Processor();
  for (let i = 0; i < 30; i++) processor.process([[new Float32Array(128).fill(.5)]]);
  assert.equal(output.length, 1); assert.equal(output[0].byteLength, 2560);
  assert.equal(new Int16Array(output[0])[0], 16383);
  console.log('PASS microphone resampling emits 80 ms PCM16 at 16 kHz');
}
