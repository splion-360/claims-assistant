// Controlled browser primitives; no real devices or network.
const vm=require('node:vm'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const src=fs.readFileSync(path.join(__dirname,'../live_demo/app.js'),'utf8').replace(/\ncreateSession\(true\);\s*$/,'\n');
function env(){
 const elements=new Map(), sockets=[], sources=[], storage=new Map();let counter=0;
 const el=()=>({textContent:'',className:'',hidden:false,style:{},dataset:{},classList:{add(){},remove(){}},querySelector(){return el()},querySelectorAll(){return[]},addEventListener(){},focus(){},showModal(){},close(){},getTracks(){return[]}});
 class WS{static OPEN=1;constructor(url){this.url=url;this.readyState=0;sockets.push(this);setImmediate(()=>{if(this.closed)return;this.readyState=1;this.onmessage?.({data:JSON.stringify({type:'ready',session_id:new URL(url).searchParams.get('session_id')})})})}send(x){this.sent??=[];this.sent.push(JSON.parse(x))}close(){this.closed=true;this.readyState=3;this.onclose?.()}}
 class Audio{constructor(){this.currentTime=10;this.destination={}}async resume(){}createBuffer(c,n,r){return {duration:n/r,getChannelData(){return new Float32Array(n)}}}createBufferSource(){const x={stopped:false,connect(){},start(at){this.startAt=at},stop(){this.stopped=true}};sources.push(x);return x}}
 const ctx=vm.createContext({console,crypto,document:{querySelector(k){if(!elements.has(k))elements.set(k,el());return elements.get(k)}},window:{location:{protocol:'http:',origin:'http://127.0.0.1:4188'},AudioContext:Audio,addEventListener(){},setInterval(){return 1},clearInterval(){}},sessionStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},navigator:{},WebSocket:WS,AudioContext:Audio,fetch:async url=>({ok:true,json:async()=>url.endsWith('/health')?{camera:{frame_interval_ms:350,frame_width:800,ideal_width:960,ideal_height:540,jpeg_quality:0.65}}:{session_id:'session-'+(++counter),has_api_key:true,state:{session_id:'session-'+counter,transcript:[],tool_activity:[]}}}),setTimeout,clearTimeout,atob:s=>Buffer.from(s,'base64').toString('binary'),btoa:s=>Buffer.from(s,'binary').toString('base64')});
 vm.runInContext(src,ctx);vm.runInContext('state={...emptyState};sessionId="initial";',ctx);return {ctx,sockets,sources,elements,run:x=>vm.runInContext(x,ctx)};
}
const tests=[];const test=(name,fn)=>tests.push([name,fn]);
test('new intake closes original socket and rejects late messages',async()=>{const e=env();await e.run('connectLive()');const old=e.sockets[0];await e.run('createSession(false)');old.onmessage({data:JSON.stringify({type:'transcript',session_id:'initial',id:'late',speaker:'Claimant',text:'Old claim'})});await e.run('sendClaimantTurn("Second claimant")');assert(old.closed);assert.equal(e.sockets.length,2);assert.equal(e.run('state.transcript.length'),1);assert.equal(old.sent,undefined);});
test('concurrent connects share a socket',async()=>{const e=env();await Promise.all([e.run('connectLive()'),e.run('connectLive()')]);assert.equal(e.sockets.length,1);e.run('disconnectLive()');});
test('reconnect uses the same claim session',async()=>{const e=env();await e.run('connectLive()');e.run('disconnectLive()');await e.run('connectLive()');assert.equal(e.sockets[0].url,e.sockets[1].url);e.run('disconnectLive()');});
test('disconnect releases camera',async()=>{const e=env();await e.run('connectLive()');e.run('let stopped=0;cameraStream={getTracks:()=>[{stop(){stopped++}}]};');e.sockets[0].close();assert.equal(e.run('stopped'),1);assert.equal(e.run('cameraStream'),null);});
test('state merge preserves final turns not yet in snapshot',()=>{const e=env();assert.equal(e.run('mergeTranscript([{id:"1",speaker:"Agent",text:"Earlier reply"}],[{id:"2",speaker:"Claimant",text:"Correction",streaming:false}]).length'),2);});
test('short answers are distinct; repeated final ID is idempotent',()=>{const e=env();e.run('upsertStreamingTurn("Claimant","No injuries",true,"a");upsertStreamingTurn("Claimant","No",true,"b");upsertStreamingTurn("Claimant","No",true,"b")');assert.equal(e.run('state.transcript.length'),2);});
test('finished and failed tools clear writing indicator',()=>{const e=env();for(const phase of ['done','error','cancelled']){e.run(`state={...emptyState,tool_activity:[{id:"1",name:"sync_claim_packet",phase:"${phase}"}]};setWriting(true);renderTeam()`);assert.equal(e.run('writing'),false);}});
test('workflow errors release busy state',async()=>{const e=env();await e.run('connectLive()');e.run('processing=true;render()');e.sockets[0].onmessage({data:JSON.stringify({type:'error',session_id:'initial',message:'Workflow failed'})});assert.equal(e.run('writing'),false);e.run('disconnectLive()');});
test('available documents remain unchecked and are labeled',()=>{const e=env();e.run('state={...emptyState,documents:[{item:"Photo",status:"available",already_provided:false}]};renderNeeded()');assert(e.elements.get('#neededList').innerHTML.includes('Photo — available'));assert(!e.elements.get('#neededList').innerHTML.includes('class="done"'));});
test('unconfirmed camera capture is labeled even without a description',()=>{const e=env();e.run('state={...emptyState,evidence_photos:[{id:"p",caption:"Wall",data_url:"data:image/jpeg;base64,AA==",confirmed:false}]};renderPinboard()');assert(e.elements.get('#pinboard').innerHTML.includes('Claim not confirmed by this image'));});
test('camera activation and stop send explicit mode changes',async()=>{
 const e=env();e.run('let tracksStopped=0;let deviceEnded;const videoTrack={stop(){tracksStopped++},addEventListener(name,fn){if(name==="ended")deviceEnded=fn}};navigator.mediaDevices={getUserMedia:async()=>({getTracks:()=>[videoTrack],getVideoTracks:()=>[videoTrack]})}');
 await e.run('startCamera()');assert.deepEqual(e.sockets[0].sent,[{type:'camera_state',enabled:true}]);
 e.run('stopCamera()');assert.deepEqual(e.sockets[0].sent.at(-1),{type:'camera_state',enabled:false});assert.equal(e.run('tracksStopped'),1);e.run('disconnectLive()');
});
test('camera device ending turns mode off',async()=>{
 const e=env();e.run('let deviceEnded;const videoTrack={stop(){},addEventListener(name,fn){if(name==="ended")deviceEnded=fn}};navigator.mediaDevices={getUserMedia:async()=>({getTracks:()=>[videoTrack],getVideoTracks:()=>[videoTrack]})}');
 await e.run('startCamera()');e.run('deviceEnded()');assert.deepEqual(e.sockets[0].sent.at(-1),{type:'camera_state',enabled:false});assert.equal(e.run('cameraStream'),null);e.run('disconnectLive()');
});
test('denied camera does not announce camera on',async()=>{
 const e=env();e.run('navigator.mediaDevices={getUserMedia:async()=>{throw new Error("Permission denied")}}');
 await e.run('startCamera()');assert.equal(e.sockets[0].sent,undefined);assert.equal(e.run('cameraStream'),null);e.run('disconnectLive()');
});
test('camera settings come from backend and socket backpressure skips frames',async()=>{
 const e=env();await e.run('connectLive()');
 await e.run('createSession(true)');await e.run('connectLive()');
 assert.equal(e.run('cameraConfig.frame_interval_ms'),350);
 assert.equal(e.run('cameraConfig.frame_width'),800);
 e.run('cameraStream={};liveSocket.bufferedAmount=600000;cameraPreview.videoWidth=640;sendFrame()');
 assert.equal(e.sockets.at(-1).sent,undefined);
 e.run('liveSocket.bufferedAmount=0;cameraPreview.videoHeight=480;frameCanvas.getContext=()=>({drawImage(){}});let jpegSettings;frameCanvas.toDataURL=(type,quality)=>{jpegSettings={type,quality};return "data:image/jpeg;base64,AA=="};sendFrame()');
 assert.equal(e.run('frameCanvas.width'),800);assert.equal(e.run('frameCanvas.height'),600);
 assert.equal(e.run('jpegSettings.quality'),0.65);
 assert.equal(e.sockets.at(-1).sent.at(-1).type,'video');
 e.run('cameraStream=null;disconnectLive()');
});
test('book inventory escapes readings and preserves unidentified copies',async()=>{
 const e=env();e.run('state.book_inventory=[{id:"book",title:"<script>bad</script>",author:null,publisher:null,observation_count:2,review_reasons:["Possible duplicate"]}];renderBookInventory()');
 const html=e.run('document.querySelector("#bookInventory").innerHTML');
 assert.ok(html.includes('&lt;script&gt;bad&lt;/script&gt;'));
 assert.ok(!html.includes('<script>'));
 assert.ok(html.includes('1 provisional copies'));
 assert.ok(html.includes('Author unreadable'));
 assert.ok(html.includes('Needs review'));
 e.run('state.book_inventory=[];renderBookInventory()');
 assert.equal(e.run('document.querySelector("#bookInventory").hidden'),true);
});
(async()=>{for(const [name,fn]of tests){await fn();console.log('PASS '+name)}console.log(`${tests.length} client regression tests passed`)})().catch(e=>{console.error(e);process.exitCode=1});
