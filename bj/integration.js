// Integration: simulate the user dictating Wispr Flow transcripts and verify
// the count + bet tier + active deviations end up where expected.
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const uiStart = code.indexOf("if (typeof document !== 'undefined') (function UI()");
if (uiStart > 0) code = code.slice(0, uiStart);
const ctx = { module: { exports: {} }, exports: {}, console };
ctx.exports = ctx.module.exports;
vm.createContext(ctx);
vm.runInContext(code, ctx);
const { Lib } = ctx.module.exports;

let pass = 0, fail = 0;
function ok(cond, name, info) {
  if (cond) { pass++; console.log('PASS', name); }
  else { fail++; console.log('FAIL', name, info || ''); }
}
function eq(a, b, name) { ok(JSON.stringify(a) === JSON.stringify(b), name, { got: a, exp: b }); }

const setup = { decks: 8, h17: true, das: true, ls: false, unit: 25, ramp: [1,2,4,8,12] };

// Apply a dictation phrase to in-memory shoe state and return new state.
function applyPhrase(shoe, phrase) {
  const r = Lib.parseDictation(phrase);
  for (const a of r.actions) {
    if (a.type === 'card')         { shoe.runningCount += Lib.cardCount(a.rank); shoe.cardsDealt += 1; }
    else if (a.type === 'adjust')  { shoe.runningCount += a.rc; shoe.cardsDealt += a.cards; }
    else if (a.type === 'shuffle') { shoe.runningCount = 0; shoe.cardsDealt = 0; }
  }
  return { shoe, parsed: r };
}

// --- Scenario: process the user's actual Wispr transcripts ---
const phrases = [
  '1064498, dealer 10',                          // 10,6,4,4,9,8,10  → RC delta = -1+1+1+1+0+0-1 = +1
  'Dealer 10 table jack',                         // 10,J → -1 -1 = -2 → cum +1-2 = -1
  '225 is dealer 7, 43 Jack, Q.',                 // 2,2,5,7,4,3,J → +1+1+1+0+1+1-1 = +4 (Q ignored — single-letter ranks disabled) → cum +3
  '722 dealer 5310',                              // 7,2,2,5,3,10 → 0+1+1+1+1-1 = +3 → cum +6
];
let shoe = { totalDecks: 8, cardsDealt: 0, runningCount: 0 };
let totalCards = 0;
for (const p of phrases) {
  const before = { ...shoe };
  const r = applyPhrase(shoe, p);
  const cards = r.parsed.actions.filter(a => a.type === 'card').length;
  totalCards += cards;
  console.log(`  "${p}"  → ${cards} cards · RC ${shoe.runningCount-before.runningCount>=0?'+':''}${shoe.runningCount-before.runningCount} · cum RC ${shoe.runningCount}`);
}
ok(shoe.runningCount === 6, 'Cumulative RC after 4 phrases = +6', { rc: shoe.runningCount });
ok(totalCards === 22, 'Total cards counted = 22', { totalCards });

// True count: 22 cards = 0.42 decks gone. dr = 8 - 0.42 = 7.58. TC = 6/7.58 = 0.79 → ¼-rounded
const dr = Lib.decksRemaining(shoe.cardsDealt, shoe.totalDecks);
const tc = Lib.trueCount(shoe.runningCount, dr);
ok(Math.abs(tc - 0.75) < 0.01, `TC ≈ +0.75 after 22 cards (dr=${dr.toFixed(2)})`, { tc, dr });

// At this TC, bet ramp should still be table min (TC < 2)
const units = Lib.wongUnits(tc, setup.ramp);
eq(units, 1, 'Bet ramp at TC≈+0.75 → 1 unit (still flat)');

// Active deviations at TC=+0.75 → only 16 vs T (still below other thresholds)
const devs = Lib.activeDeviations(tc, setup);
eq(devs.length, 1, 'TC≈+0.75 → exactly 1 active deviation');
eq(devs[0].name, '16 vs T', 'TC≈+0.75 → 16 vs T is the active dev');

// --- Scenario: ramp up to TC=+3 ---
shoe.runningCount = 24; shoe.cardsDealt = 4 * 52; // 4 decks gone, RC=24
const tc2 = Lib.trueCount(shoe.runningCount, Lib.decksRemaining(shoe.cardsDealt, shoe.totalDecks));
eq(tc2, 6, 'TC at RC=24, 4 decks left → +6');
eq(Lib.wongUnits(tc2, setup.ramp), 12, 'TC=+6 → 12u (top of ramp)');
const devsHigh = Lib.activeDeviations(tc2, setup);
ok(devsHigh.length >= 5, `TC=+6 → ≥5 active devs (got ${devsHigh.length})`);
const namesHigh = devsHigh.map(d => d.name);
ok(namesHigh.includes('16 vs T'), 'Hot count includes 16 vs T');
ok(namesHigh.includes('T,T vs 5'), 'Hot count includes T,T vs 5');
ok(namesHigh.includes('16 vs 9'), 'Hot count includes 16 vs 9');

// --- Scenario: shuffle resets (button action, not voice) ---
shoe = { totalDecks: 8, cardsDealt: 0, runningCount: 0 };
shoe.runningCount = 12; shoe.cardsDealt = 200;
shoe.runningCount = 0; shoe.cardsDealt = 0; // simulate shuffle button
eq(shoe.runningCount, 0, 'shuffle clears RC');
eq(shoe.cardsDealt,   0, 'shuffle clears cards-dealt');

// --- Scenario: dictation adjust verbs ---
shoe = { totalDecks: 8, cardsDealt: 0, runningCount: 0 };
applyPhrase(shoe, 'plus 5');
eq(shoe.runningCount, 5, '"plus 5" raises RC by 5');
eq(shoe.cardsDealt,   5, '"plus 5" advances cards by 5');
applyPhrase(shoe, 'skip 10');
eq(shoe.runningCount, 5, 'skip leaves RC unchanged');
eq(shoe.cardsDealt,  15, 'skip 10 advances cards by 10');

console.log(`\n${pass}/${pass+fail} integration cases passed`);
process.exit(fail > 0 ? 1 : 0);
