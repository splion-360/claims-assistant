// Controlled browser primitives; no real devices or network.
const vm=require('node:vm'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const src=fs.readFileSync(path.join(__dirname,'../live_demo/inventory.js'),'utf8')
  .replace(/\nrefresh\(\);\s*\nwindow\.setInterval[^\n]*\n?$/,'\n');
function env(state){
 const elements=new Map();
 const el=()=>({textContent:'',innerHTML:'',addEventListener(){}});
 const ctx=vm.createContext({console,
  document:{querySelector(k){if(!elements.has(k))elements.set(k,el());return elements.get(k)}},
  window:{location:{origin:'http://127.0.0.1:4177',search:'?session=abcdef1234'},setInterval(){return 1}},
  encodeURIComponent,
  URLSearchParams,Date,Number,String,
  sessionStorage:{getItem:()=>null},
  fetch:async()=>({ok:true,status:200,json:async()=>({state})})});
 vm.runInContext(src,ctx);
 return {ctx,run:x=>vm.runInContext(x,ctx),html:k=>elements.get(k)?.innerHTML??'',text:k=>elements.get(k)?.textContent??''};
}
const tests=[];const test=(name,fn)=>tests.push([name,fn]);

test('a priced book shows its amount and links the source',async()=>{
 const e=env({book_inventory:[{id:'b1',title:'THE GOOD PEOPLE',author:'HANNAH KENT',observation_count:2,
   catalogue:{isbn:'9780316243933'},spine_height_cm:19.5,spine_thickness_cm:3.2,
   valuation:{needs_appraisal:false,replacement:{amount:17.99,currency:'USD',source:'Google Books',url:'https://books.example/buy',converted:false}}}],item_inventory:[]});
 await e.run('refresh()');
 const html=e.html('#books');
 assert.ok(html.includes('USD 17.99'));
 assert.ok(html.includes('https://books.example/buy'));
 assert.ok(html.includes('19.5')&&html.includes('3.2'));
});
test('an unmeasured, unpriced book says so rather than showing zero',async()=>{
 const e=env({book_inventory:[{id:'b1',title:'NO PRICE',observation_count:1}],item_inventory:[]});
 await e.run('refresh()');
 const html=e.html('#books');
 assert.ok(html.includes('not measured'));
 assert.ok(html.includes('not priced yet'));
 assert.ok(!html.includes('0.00'));
});
test('a converted figure names the market it came from',async()=>{
 const e=env({book_inventory:[{id:'b1',title:'X',observation_count:1,
   valuation:{needs_appraisal:false,replacement:{amount:13.6,currency:'USD',source:'Google Books',url:'',converted:true,converted_from:'GBP'}}}],item_inventory:[]});
 await e.run('refresh()');
 assert.ok(e.html('#books').includes('converted from GBP'));
});
test('objects list their category, readings and review reasons',async()=>{
 const e=env({book_inventory:[],item_inventory:[{id:'i1',category:'lighting',description:'brass floor lamp',
   material:'brass',brand_model:'Anglepoise 1227',observation_count:2,needs_review:true,
   review_reasons:['Object association unverified; possible duplicate']}]});
 await e.run('refresh()');
 const html=e.html('#items');
 assert.ok(html.includes('lighting'));
 assert.ok(html.includes('Anglepoise 1227'));
 assert.ok(html.includes('possible duplicate'));
 assert.ok(html.includes('review'));
});
test('framed art waits for an appraiser instead of showing a price',async()=>{
 const e=env({book_inventory:[],item_inventory:[{id:'i2',category:'art',description:'framed portrait',
   observation_count:1,needs_appraisal:true,review_reasons:[]}]});
 await e.run('refresh()');
 const html=e.html('#items');
 assert.ok(html.includes('awaiting appraisal'));
 assert.ok(html.includes('appraisal'));
});
test('an object with no provider says no retail source is configured',async()=>{
 const e=env({book_inventory:[],item_inventory:[{id:'i3',category:'seating',description:'armchair',
   observation_count:1,needs_appraisal:false,review_reasons:[]}]});
 await e.run('refresh()');
 assert.ok(e.html('#items').includes('no retail source configured'));
});
test('each line shows a crop of the sighting it came from',async()=>{
 const e=env({book_inventory:[{id:'b1',title:'A BOOK',observation_count:1,observations:[{id:'obs-1'}]}],
   item_inventory:[{id:'i1',category:'lighting',description:'lamp',observation_count:1,
     observation_ids:['obs-2'],review_reasons:[]}]});
 await e.run('refresh()');
 assert.ok(e.html('#books').includes('/api/sessions/abcdef1234/books/evidence/obs-1'));
 assert.ok(e.html('#items').includes('/api/sessions/abcdef1234/items/evidence/obs-2'));
});
test('a line with no sighting shows no broken image',async()=>{
 const e=env({book_inventory:[{id:'b1',title:'A BOOK',observation_count:0,observations:[]}],item_inventory:[]});
 await e.run('refresh()');
 assert.ok(!e.html('#books').includes('<img'));
});
test('every object offers the three review decisions',async()=>{
 const e=env({book_inventory:[],item_inventory:[{id:'i1',category:'seating',description:'chair',
   observation_count:1,needs_review:true,review_reasons:['possible duplicate'],observation_ids:[]}]});
 await e.run('refresh()');
 const html=e.html('#items');
 for (const action of ['confirm','appraise','exclude']) {
   assert.ok(html.includes(`data-action="${action}"`),`missing ${action}`);
 }
 assert.ok(html.includes('data-item="i1"'));
});
test('settling a review posts the decision and redraws from the answer',async()=>{
 const e=env({book_inventory:[],item_inventory:[{id:'i1',category:'seating',description:'chair',
   observation_count:1,needs_review:true,review_reasons:['possible duplicate'],observation_ids:[]}]});
 await e.run('refresh()');
 const sent=[];
 e.ctx.fetch=async(url,options)=>{sent.push({url,body:JSON.parse(options.body)});
   return {ok:true,status:200,json:async()=>({state:{item_inventory:[{id:'i1',category:'seating',
     description:'chair',observation_count:1,needs_review:false,review_reasons:[],observation_ids:[]}]}})}};
 await e.run('settleItem("abcdef1234","i1","confirm",{disabled:false})');
 assert.equal(sent.length,1);
 assert.ok(sent[0].url.endsWith('/items/i1/review'));
 assert.deepEqual(sent[0].body,{action:'confirm'});
 // The redraw comes from the server's answer, not from a guess.
 assert.ok(e.html('#items').includes('recorded'));
});
test('model-supplied text is escaped',async()=>{
 const e=env({book_inventory:[{id:'b1',title:'<script>bad</script>',observation_count:1}],
   item_inventory:[{id:'i1',category:'other',description:'<img onerror=x>',observation_count:1,review_reasons:[]}]});
 await e.run('refresh()');
 assert.ok(!e.html('#books').includes('<script>bad'));
 assert.ok(!e.html('#items').includes('<img onerror'));
 assert.ok(e.html('#books').includes('&lt;script&gt;'));
});
test('empty tables say nothing is recorded yet',async()=>{
 const e=env({book_inventory:[],item_inventory:[]});
 await e.run('refresh()');
 assert.ok(e.html('#books').includes('No books recorded yet'));
 assert.ok(e.html('#items').includes('No non-book contents recorded yet'));
});
test('counts report what was recorded and identified',async()=>{
 const e=env({book_inventory:[{id:'b1',title:'A',observation_count:1},{id:'b2',observation_count:1}],
   item_inventory:[{id:'i1',category:'rug',description:'rug',observation_count:1,review_reasons:[]}]});
 await e.run('refresh()');
 assert.ok(e.text('#bookCount').includes('2 recorded'));
 assert.ok(e.text('#bookCount').includes('1 identified'));
 assert.ok(e.text('#itemCount').includes('1 recorded'));
});

(async()=>{let failed=0;
 for(const [name,fn] of tests){try{await fn();console.log('PASS '+name)}catch(error){failed++;console.error('FAIL '+name+'\n  '+error.message)}}
 console.log(`${tests.length-failed} inventory regression tests passed`);
 process.exit(failed?1:0);})();
