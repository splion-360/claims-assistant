// Produce 80 ms PCM16 frames at 16 kHz from the device's actual sample rate.
class MicrophoneCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.samples = [];
    this.frameSize = Math.round(sampleRate * 0.08);
  }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    for (const value of input) this.samples.push(value);
    while (this.samples.length >= this.frameSize) {
      const frame = this.samples.splice(0, this.frameSize);
      const pcm = new Int16Array(1280);
      for (let i = 0; i < pcm.length; i++) {
        const start = Math.floor(i * frame.length / pcm.length);
        const end = Math.max(start + 1, Math.floor((i + 1) * frame.length / pcm.length));
        let sum = 0;
        for (let j = start; j < end; j++) sum += frame[j];
        const value = Math.max(-1, Math.min(1, sum / (end - start)));
        pcm[i] = value < 0 ? value * 32768 : value * 32767;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}
registerProcessor('microphone-capture', MicrophoneCapture);
