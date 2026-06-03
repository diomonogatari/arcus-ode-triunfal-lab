// arcus_solver — native nanoGPT forward for the Arcus "ode.pt" model, for EXHAUSTIVE,
// parallel trigger discovery (the right tool for a CPU-only box). Reimplements the exact
// forward used by probe_campos.py (LayerNorm bias=False eps=1e-5, exact-erf GELU, tied lm_head),
// validated by reproducing the decoy greedily before any sweep is trusted.

use ndarray::prelude::*;
use ndarray::{s, Array1, Array2};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct TensorMeta { name: String, shape: Vec<usize>, offset: usize, n: usize }
#[derive(Deserialize)]
struct Config { vocab_size: usize, block_size: usize, n_layer: usize, n_head: usize, n_embd: usize }
#[derive(Deserialize)]
struct Manifest { config: Config, tensors: Vec<TensorMeta> }

struct Layer { ln1: Array1<f32>, ln2: Array1<f32>, c_attn: Array2<f32>, c_proj: Array2<f32>, fc: Array2<f32>, proj: Array2<f32> }
struct Model { wte: Array2<f32>, wpe: Array2<f32>, layers: Vec<Layer>, lnf: Array1<f32>, lm_head: Array2<f32>, nh: usize, c: usize, vocab: usize }

fn layernorm(x: &Array2<f32>, w: &Array1<f32>) -> Array2<f32> {
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let n = row.len() as f32;
        let mean = row.sum() / n;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for (j, v) in row.iter_mut().enumerate() { *v = (*v - mean) * inv * w[j]; }
    }
    out
}
fn gelu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|z| 0.5 * z * (1.0 + libm::erff(z / std::f32::consts::SQRT_2)))
}
fn ln1d(x: &Array1<f32>, w: &Array1<f32>) -> Array1<f32> {
    let n = x.len() as f32; let mean = x.sum() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    Array1::from_iter(x.iter().enumerate().map(|(j, v)| (v - mean) * inv * w[j]))
}
fn gelu1d(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|z| 0.5 * z * (1.0 + libm::erff(z / std::f32::consts::SQRT_2)))
}

impl Model {
    fn attn(&self, h: &Array2<f32>, l: &Layer) -> Array2<f32> {
        let t = h.nrows(); let c = self.c; let hd = c / self.nh;
        let qkv = h.dot(&l.c_attn.t());                 // [T, 3C]
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = Array2::<f32>::zeros((t, c));
        for head in 0..self.nh {
            let qo = head * hd; let ko = c + head * hd; let vo = 2 * c + head * hd;
            let q = qkv.slice(s![.., qo..qo + hd]);
            let k = qkv.slice(s![.., ko..ko + hd]);
            let v = qkv.slice(s![.., vo..vo + hd]);
            let mut scores = q.dot(&k.t());             // [T, T]
            scores *= scale;
            for i in 0..t {
                // causal mask + softmax over j<=i
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=i { if scores[[i, j]] > mx { mx = scores[[i, j]]; } }
                let mut sum = 0.0;
                for j in 0..=i { let e = (scores[[i, j]] - mx).exp(); scores[[i, j]] = e; sum += e; }
                for j in 0..=i { scores[[i, j]] /= sum; }
                for j in (i + 1)..t { scores[[i, j]] = 0.0; }
            }
            let oh = scores.dot(&v);                     // [T, hd]
            out.slice_mut(s![.., qo..qo + hd]).assign(&oh);
        }
        out.dot(&l.c_proj.t())
    }

    // last-position logits for greedy
    fn forward_last(&self, ids: &[usize]) -> Array1<f32> {
        let t = ids.len();
        let mut x = Array2::<f32>::zeros((t, self.c));
        for (i, &tok) in ids.iter().enumerate() {
            let e = &self.wte.row(tok) + &self.wpe.row(i);
            x.row_mut(i).assign(&e);
        }
        for l in &self.layers {
            let a = self.attn(&layernorm(&x, &l.ln1), l);
            x = &x + &a;
            let m = gelu(&layernorm(&x, &l.ln2).dot(&l.fc.t())).dot(&l.proj.t());
            x = &x + &m;
        }
        let xf = layernorm(&x, &self.lnf);
        self.lm_head.dot(&xf.row(t - 1).to_owned())     // [vocab]
    }

    // single-token forward with KV cache (ck/cv are per-layer flattened [T*C] buffers)
    fn step(&self, tok: usize, pos: usize, ck: &mut [Vec<f32>], cv: &mut [Vec<f32>]) -> Array1<f32> {
        let c = self.c; let hd = c / self.nh; let scale = 1.0 / (hd as f32).sqrt();
        let mut x = &self.wte.row(tok) + &self.wpe.row(pos);   // Array1 [C]
        for (li, l) in self.layers.iter().enumerate() {
            let h = ln1d(&x, &l.ln1);
            let qkv = l.c_attn.dot(&h);                         // [3C]
            ck[li].extend_from_slice(qkv.slice(s![c..2 * c]).as_slice().unwrap());
            cv[li].extend_from_slice(qkv.slice(s![2 * c..3 * c]).as_slice().unwrap());
            let len = ck[li].len() / c;
            let mut attout = vec![0f32; c];
            for head in 0..self.nh {
                let off = head * hd;
                let mut scores = vec![0f32; len]; let mut mx = f32::NEG_INFINITY;
                for j in 0..len {
                    let mut s = 0.0; for d in 0..hd { s += qkv[off + d] * ck[li][j * c + off + d]; }
                    s *= scale; scores[j] = s; if s > mx { mx = s; }
                }
                let mut sum = 0.0; for j in 0..len { let e = (scores[j] - mx).exp(); scores[j] = e; sum += e; }
                for d in 0..hd {
                    let mut acc = 0.0; for j in 0..len { acc += scores[j] * cv[li][j * c + off + d]; }
                    attout[off + d] = acc / sum;
                }
            }
            x = &x + &l.c_proj.dot(&Array1::from(attout));
            let h2 = ln1d(&x, &l.ln2);
            x = &x + &l.proj.dot(&gelu1d(&l.fc.dot(&h2)));
        }
        self.lm_head.dot(&ln1d(&x, &self.lnf))
    }

    fn greedy_cached(&self, prefix: &[usize], n: usize) -> (Vec<usize>, f32) {
        let nl = self.layers.len();
        let mut ck = vec![Vec::<f32>::new(); nl]; let mut cv = vec![Vec::<f32>::new(); nl];
        let mut pos = 0usize; let mut last = Array1::<f32>::zeros(self.vocab);
        for &t in prefix { last = self.step(t, pos, &mut ck, &mut cv); pos += 1; }
        let mut out = Vec::new(); let mut conf = 0.0f32; let mut steps = 0;
        for _ in 0..n {
            let mut mi = 0usize; let mut mv = f32::NEG_INFINITY;
            for (j, &v) in last.iter().enumerate() { if v > mv { mv = v; mi = j; } }
            let mut sum = 0.0; for &v in last.iter() { sum += (v - mv).exp(); }
            conf += 1.0 / sum; steps += 1; out.push(mi);
            if mi == 125 { break; }
            last = self.step(mi, pos, &mut ck, &mut cv); pos += 1;
        }
        (out, if steps > 0 { conf / steps as f32 } else { 0.0 })
    }

    fn greedy(&self, prefix: &[usize], n: usize) -> (Vec<usize>, f32) {
        let mut ids = prefix.to_vec();
        let mut out = Vec::new(); let mut conf = 0.0f32; let mut steps = 0;
        for _ in 0..n {
            let lg = self.forward_last(&ids);
            // argmax + softmax-max
            let mut mi = 0usize; let mut mv = f32::NEG_INFINITY;
            for (j, &v) in lg.iter().enumerate() { if v > mv { mv = v; mi = j; } }
            let mut sum = 0.0; for &v in lg.iter() { sum += (v - mv).exp(); }
            conf += 1.0 / sum; steps += 1;
            ids.push(mi); out.push(mi);
            if mi == 125 { break; }                     // '}'
        }
        (out, if steps > 0 { conf / steps as f32 } else { 0.0 })
    }
}

fn load(dir: &str) -> Model {
    let man: Manifest = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap()).unwrap();
    let bytes = std::fs::read(format!("{dir}/weights.bin")).unwrap();
    let get = |m: &TensorMeta| -> Vec<f32> {
        let mut v = Vec::with_capacity(m.n);
        for k in 0..m.n { let o = m.offset + k * 4; v.push(f32::from_le_bytes([bytes[o], bytes[o+1], bytes[o+2], bytes[o+3]])); }
        v
    };
    let mut map: HashMap<String, &TensorMeta> = HashMap::new();
    for t in &man.tensors { map.insert(t.name.clone(), t); }
    let a2 = |name: &str| { let m = map[name]; Array2::from_shape_vec((m.shape[0], m.shape[1]), get(m)).unwrap() };
    let a1 = |name: &str| { let m = map[name]; Array1::from_vec(get(m)) };
    let c = man.config.clone_n_embd();
    let mut layers = Vec::new();
    for i in 0..man.config.n_layer {
        layers.push(Layer {
            ln1: a1(&format!("transformer.h.{i}.ln_1.weight")),
            ln2: a1(&format!("transformer.h.{i}.ln_2.weight")),
            c_attn: a2(&format!("transformer.h.{i}.attn.c_attn.weight")),
            c_proj: a2(&format!("transformer.h.{i}.attn.c_proj.weight")),
            fc: a2(&format!("transformer.h.{i}.mlp.c_fc.weight")),
            proj: a2(&format!("transformer.h.{i}.mlp.c_proj.weight")),
        });
    }
    let lm = if map.contains_key("lm_head.weight") { a2("lm_head.weight") } else { a2("transformer.wte.weight") };
    Model { wte: a2("transformer.wte.weight"), wpe: a2("transformer.wpe.weight"), layers,
            lnf: a1("transformer.ln_f.weight"), lm_head: lm, nh: man.config.n_head, c, vocab: man.config.vocab_size }
}

impl Config { fn clone_n_embd(&self) -> usize { self.n_embd } }

// ---- tokenizer (byte-level + greedy specials) ----
fn encode(s: &str) -> Vec<usize> {
    let specials = [("<|fernando_pessoa|>", 256usize), ("<|alberto_caeiro|>", 257), ("<|ricardo_reis|>", 258),
                    ("<|bernardo_soares|>", 259), ("_", 260), ("{", 261)];
    let b = s.as_bytes(); let mut i = 0; let mut out = Vec::new();
    'outer: while i < b.len() {
        for (tok, id) in specials.iter() {
            let tb = tok.as_bytes();
            if b[i..].starts_with(tb) { out.push(*id); i += tb.len(); continue 'outer; }
        }
        out.push(b[i] as usize); i += 1;
    }
    out
}
fn decode(ids: &[usize]) -> String {
    let mut bytes = Vec::new(); let mut s = String::new();
    let names: HashMap<usize, &str> = [(256,"<|fernando_pessoa|>"),(257,"<|alberto_caeiro|>"),(258,"<|ricardo_reis|>"),(259,"<|bernardo_soares|>"),(260,"_"),(261,"{")].into_iter().collect();
    for &t in ids {
        if t < 256 { bytes.push(t as u8); }
        else { if !bytes.is_empty() { s.push_str(&String::from_utf8_lossy(&bytes)); bytes.clear(); } s.push_str(names[&t]); }
    }
    if !bytes.is_empty() { s.push_str(&String::from_utf8_lossy(&bytes)); }
    s
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("validate");
    let m = load(".");
    eprintln!("model loaded: {} layers, C={}, vocab={}", m.layers.len(), m.c, m.vocab);

    if mode == "validate" {
        // must reproduce the decoy if the forward is correct
        let p = encode("<|alvaro_de_campos|>flag{");
        let (out, c) = m.greedy(&p, 30);
        println!("decoy check: conf={:.3}\n  {:?}", c, decode(&out));
        let p2 = encode("flag{");
        let (o2, c2) = m.greedy(&p2, 20);
        println!("flag{{ greedy: conf={:.3}  {:?}", c2, decode(&o2));
        // KV-cache parity: must match the naive forward exactly
        let (oc, cc) = m.greedy_cached(&p, 30);
        println!("decoy (KV-cache): conf={:.3}\n  {:?}", cc, decode(&oc));
        return;
    }

    if mode == "sweep1" || mode == "sweep2" {
        let flagb = encode("flag{");
        // build prefix list
        let prefixes: Vec<Vec<usize>> = if mode == "sweep1" {
            (0..m.vocab).map(|t| vec![t]).collect()
        } else {
            let mut v = Vec::new();
            for a in 0..m.vocab { for b in 0..m.vocab { v.push(vec![a, b]); } }
            v
        };
        eprintln!("sweeping {} prefixes (+ \"flag{{\"), greedy 48 each...", prefixes.len());
        let hits: Vec<String> = prefixes.par_iter().filter_map(|pre| {
            let mut input = pre.clone(); input.extend_from_slice(&flagb);
            let (out, conf) = m.greedy_cached(&input, 32);
            let has_close = out.contains(&125);
            let has_us = out.iter().any(|&t| t == 95 || t == 260);
            let has_digit = out.iter().any(|&t| (48..=57).contains(&t));
            if has_close || has_us || has_digit {
                let txt = decode(&out);
                let decoy = txt.contains("Hup") || txt.contains("z-z") || txt.contains("He-");
                if !decoy {
                    return Some(format!("pre={:?} conf={:.2} close={} us={} dig={} :: {:?}",
                        decode(pre), conf, has_close, has_us, has_digit, &txt[..txt.len().min(90)]));
                }
            }
            None
        }).collect();
        eprintln!("=== {} interesting (non-decoy) completions ===", hits.len());
        for h in &hits { println!("{h}"); }
    }
}
