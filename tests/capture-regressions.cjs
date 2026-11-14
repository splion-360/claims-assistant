// Controlled browser primitives for the phone capture page; no real devices or network.
const vm=require('node:vm'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const src=fs.readFileSync(path.join(__dirname,'../live_demo/ar.js'),'utf8')
  +'\n'+fs.readFileSync(path.join(__dirname,'../live_demo/capture.js'),'utf8');
const CONFIG={frame_interval_ms:200,frame_width:800,ideal_width:1280,ideal_height:720,jpeg_quality:0.8,
  depth_grid_width:8,depth_grid_height:4};

function env({arSupported=false}={}){
 const elements=new Map(),sockets=[];
 const el=()=>({textContent:'',value:'',hidden:false,disabled:false,srcObject:null,videoWidth:0,videoHeight:0,
   classList:{add(){},remove(){},toggle(){}},addEventListener(){},getContext:()=>({drawImage(){},putImageData(){}}),
   toDataURL:()=>'data:image/jpeg;base64,QUJD'});
 class WS{static OPEN=1;constructor(url){this.url=url;this.readyState=1;this.sent=[];sockets.push(this)}
   send(x){this.sent.push(JSON.parse(x))}close(){this.readyState=3;this.onclose?.()}}
 const ctx=vm.createContext({console,assert,
  document:{querySelector(k){if(!elements.has(k))elements.set(k,el());return elements.get(k)},
            createElement:()=>el(),addEventListener(){},body:{}},
  window:{location:{protocol:'https:',host:'demo.ngrok-free.dev'},addEventListener(){},setInterval:()=>1,clearInterval(){},
          XRWebGLBinding:arSupported?function(){}:undefined},
  navigator:{xr:arSupported?{isSessionSupported:async()=>true}:undefined},
  XRWebGLBinding:arSupported?function(){}:undefined,WebSocket:WS,
  btoa:s=>Buffer.from(s,'binary').toString('base64'),Uint16Array,Uint8Array,Uint8ClampedArray,
  ImageData:function(w,h){this.data=new Uint8ClampedArray(w*h*4)},Number,Math,JSON,String,encodeURIComponent});
 vm.runInContext(src,ctx);
 return {ctx,sockets,elements,run:x=>vm.runInContext(x,ctx)};
}
const tests=[];const test=(name,fn)=>tests.push([name,fn]);

test('a phone with AR support sweeps in AR rather than the plain camera',async()=>{
 const e=env({arSupported:true});
 let started=0;
 // Replacing the session starter on the context is what rebinds it for the page.
 e.ctx.startArSession=async()=>{started++;return {end:async()=>{}}};
 e.run(`cameraConfig=${JSON.stringify(CONFIG)};socket=new WebSocket("wss://x/ws/capture");
        navigator.mediaDevices={getUserMedia:async()=>{throw new Error("plain camera must not be used")}}`);
 await e.run('startCamera()');
 assert.equal(started,1);
 assert.deepEqual(e.sockets[0].sent,[{type:'camera_state',enabled:true}]);
 assert.ok(e.run('document.querySelector("#status").textContent').includes('AR'));
});
test('a phone without AR support falls back to the plain camera',async()=>{
 const e=env({arSupported:false});
 e.run(`cameraConfig=${JSON.stringify(CONFIG)};socket=new WebSocket("wss://x/ws/capture");
        let opened=0;const track={stop(){},addEventListener(){}};
        navigator.mediaDevices={getUserMedia:async()=>{opened++;return {getTracks:()=>[track],getVideoTracks:()=>[track]}}}`);
 await e.run('startCamera()');
 assert.equal(e.run('opened'),1);
 assert.deepEqual(e.sockets[0].sent,[{type:'camera_state',enabled:true}]);
});
test('a failed AR session falls back instead of ending the sweep',async()=>{
 const e=env({arSupported:true});
 e.ctx.startArSession=async()=>{throw new Error("denied")};
 e.run(`cameraConfig=${JSON.stringify(CONFIG)};socket=new WebSocket("wss://x/ws/capture");
        let opened=0;const track={stop(){},addEventListener(){}};
        navigator.mediaDevices={getUserMedia:async()=>{opened++;return {getTracks:()=>[track],getVideoTracks:()=>[track]}}}`);
 await e.run('startCamera()');
 assert.equal(e.run('opened'),1);
});
test('stopping ends the AR session and tells the server the camera is off',async()=>{
 const e=env({arSupported:true});
 let ended=0;
 e.ctx.startArSession=async()=>({end:async()=>{ended++}});
 e.run(`cameraConfig=${JSON.stringify(CONFIG)};socket=new WebSocket("wss://x/ws/capture")`);
 await e.run('startCamera()');
 e.run('stopCamera()');
 assert.equal(ended,1);
 assert.deepEqual(e.sockets[0].sent.at(-1),{type:'camera_state',enabled:false});
});
test('an AR frame carries depth, intrinsics and the camera pose',()=>{
 const e=env({arSupported:true});
 const message=e.run(`(() => {
   const gl={bindFramebuffer(){},framebufferTexture2D(){},readPixels(){},createFramebuffer(){},FRAMEBUFFER:1,COLOR_ATTACHMENT0:2,TEXTURE_2D:3,RGBA:4,UNSIGNED_BYTE:5};
   const view={camera:{width:16,height:9},projectionMatrix:[2,0,0,0, 0,2,0,0, 0,0,-1,-1, 0,0,-0.2,0],
     transform:{matrix:new Array(16).fill(0).map((_,i)=>i)}};
   const depthInfo={getDepthInMeters:()=>1.25};
   return arFrameMessage(gl,{},document.createElement("canvas"),{},view,depthInfo,${JSON.stringify(CONFIG)});
 })()`);
 assert.equal(message.type,'video');
 assert.equal(message.depth.width,CONFIG.depth_grid_width);
 assert.equal(message.depth.pose.length,16);
 assert.ok(message.depth.fx>0&&message.depth.fy>0);
 // 1.25 m encodes as 1250 mm in the little-endian uint16 grid.
 const bytes=Buffer.from(message.depth.mm,'base64');
 assert.equal(bytes[0]|(bytes[1]<<8),1250);
});

(async()=>{let failed=0;
 for(const [name,fn] of tests){try{await fn();console.log('PASS '+name)}catch(error){failed++;console.error('FAIL '+name+'\n  '+error.message)}}
 console.log(`${tests.length-failed} capture regression tests passed`);
 process.exit(failed?1:0);})();
