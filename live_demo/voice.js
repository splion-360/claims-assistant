// Speech media only; app.js owns the session and the WebSocket.
class ClaimVoice {
  constructor({ send, status, error }) {
    this.send = send;
    this.status = status;
    this.error = error;
    this.enabled = false;
    this.ready = false;
    this.serial = 0;
    this.sources = new Set();
    this.nextTime = 0;
    this.epoch = null;
  }
  async start() {
    if (this.enabled) return;
    this.enabled = true;
    const serial = ++this.serial;
    this.status('Connecting voice');
    try {
      this.context = new AudioContext();
      await this.context.resume();
      await this.context.audioWorklet.addModule('/audio-worklet.js');
      if (serial !== this.serial) return;
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      if (serial !== this.serial) { stream.getTracks().forEach(t => t.stop()); return; }
      this.stream = stream;
      stream.getAudioTracks().forEach(t => t.addEventListener('ended', () => this.stop(), { once: true }));
      this.input = this.context.createMediaStreamSource(stream);
      this.worklet = new AudioWorkletNode(this.context, 'microphone-capture');
      this.worklet.port.onmessage = ({ data }) => {
        if (!this.ready || serial !== this.serial) return;
        let binary = '';
        for (const byte of new Uint8Array(data)) binary += String.fromCharCode(byte);
        this.send({ type: 'audio', data: btoa(binary) });
      };
      this.input.connect(this.worklet);
      // Worklet emits silence at its output; keep it attached so it runs.
      this.worklet.connect(this.context.destination);
      this.send({ type: 'voice_state', enabled: true });
      this.timeout = setTimeout(() => {
        if (!this.ready && serial === this.serial) {
          this.error('Voice connection timed out. Try again or type instead.');
          this.stop();
        }
      }, 15000);
    } catch (error) {
      if (serial !== this.serial) return;
      this.error(error.name === 'NotAllowedError' ? 'Microphone permission denied. Allow access or type instead.' : 'Voice could not start. Check microphone support and try again.');
      this.stop();
    }
  }
  setReady(enabled) {
    if (!this.enabled) return;
    if (!enabled) { this.stop(false); return; }
    clearTimeout(this.timeout);
    this.ready = true;
    this.status('Listening');
  }
  interrupt() {
    this.epoch = null;
    for (const source of this.sources) { try { source.stop(); } catch (_) {} }
    this.sources.clear();
    this.nextTime = this.context?.currentTime || 0;
  }
  begin(epoch) {
    this.interrupt();
    if (this.enabled && this.ready) this.epoch = epoch;
  }
  play(message) {
    if (!this.enabled || !this.ready || this.epoch !== message.epoch) return;
    const bytes = Uint8Array.from(atob(message.data), c => c.charCodeAt(0));
    if (!bytes.length || bytes.length % 2) return;
    const view = new DataView(bytes.buffer);
    const buffer = this.context.createBuffer(1, bytes.length / 2, 24000);
    const samples = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(i * 2, true) / 32768;
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    const start = Math.max(this.context.currentTime, this.nextTime);
    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
    source.start(start);
    this.nextTime = start + buffer.duration;
  }
  stop(notify = true) {
    const wasEnabled = this.enabled;
    this.enabled = this.ready = false;
    this.serial++;
    clearTimeout(this.timeout);
    this.interrupt();
    this.worklet?.disconnect();
    this.input?.disconnect();
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = this.input = this.worklet = null;
    const context = this.context;
    this.context = null;
    if (context) context.close().catch(() => {});
    if (notify && wasEnabled) this.send({ type: 'voice_state', enabled: false });
    this.status('Voice off');
  }
}
window.ClaimVoice = ClaimVoice;
