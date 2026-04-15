# Hariç Tutulan Parametreler — Kanıt ve Gerekçe

Bu dokuman, IAMS optimizer sweep yuzeyinden kaldirilan veya kaynak kodda
hardwired hale getirilen parametrelerin gerekcelerini ve kanitlarini tutar.
Sonra "neden bu parametre uzaya alinmadi?" sorusu gelirse cevap buradadir.

## ⚠ Baseline Metodoloji Uyarisi

**Onemli:** Bu dokumandaki tum Δwer_pp olcumleri **SafeScribeAI'nin tuned
baseline'ina** gorecelidir — yani `mobile_preprocessing` AKTIF + custom
transcription parametreleri (vad_filter=true, condition_on_previous_text=false,
log_prob_threshold=-0.8 vb.). **Pristine asrbench default'una gore DEGIL.**

Bu sunu ifade eder:

- Bir parametrenin "0 etki" olmasi, SADECE o tuned rejimde gecerlidir.
- Pristine default (preproc=off, tum fw_default degerleri, sequential mod)
  altinda ayni parametre farkli davranabilir.
- Gelecekte bir parametreyi hariç tutmadan once, onun **pristine asrbench
  default** uzerinde de calistirilmasi ve sinyal verip vermedigine bakilmasi
  gerekir.

Bu dokumandaki exclusion'lar bu yuzden **muhafazakar** tutuluyor: sadece
**iki farkli tuned baseline'a gore hepsi sifir** olan parametreler hardcoded
oldu; tek merkezden olculenler (bkz. loudnorm_linear) **RESTORED** olarak
sweep'te.

Kaynak logu: **SafeScribeAI/scripts/wer_results_log.jsonl**, phase1_s1
sweep, 2026-04-14 07:17 itibariyle. Sweep rejimi:

- **Modeller:** Systran/faster-whisper-large-v3 (fp16, batch=1, temp=0.0)
- **Datasetler:** LibriSpeech test-clean (EN), FLEURS tr_tr test (TR),
  MediaSpeech TR (gurultulu medya yayinlari). 15 dakikalik kesimler.
- **Baseline:** phase0_s1 — fix edilmis transcription/vad parametreleri +
  default `mobile_preprocessing` bloku (TUNED — pristine DEGIL).
- **Her sweep:** baseline'dan tek parametre oynatilir, phase1 WER olculur,
  `Δwer_pp = (wer - baseline_wer) × 100`.
- **Kosum sayisi:** her (parametre, deger) icin en az 6 run (3 dataset × 2
  independent baselines).

## Ozet

| Parametre | Durum | Kapsam | max\|Δwer_pp\| | Karar |
|---|---|---|---|---|
| `alimiter release_ms` | **HARDCODED** | `150` vs default `50` | 0.000 | ffmpeg_pipeline.py'da sabit 50. Zaten sweep'te degildi; test onayladi. |
| `alimiter attack_ms` | HARDCODED | `2`, `20` vs default `5` | 2.846 | Aktif, ama sweep'te degil — tuning faydasinin maliyeti karsilamadigi yerlerden. |
| `loudnorm tp` | HARDCODED | `-0.5`, `-3.0` vs default `-1.5` | 2.388 | Aktif, ama EBU R128 streaming default yeterli. |
| `preprocess.loudnorm_linear` | **RESTORED (koşullu)** | `true` vs default `false` | 0.000 @ lufs=-16 | Merkezde 0 etki, uçlarda belirsiz; L5 ablation'a birakildi. |

## Kanit Tablolari

### 1) `loudnorm_linear` — merkez olcumu sifir, ama UCLAR test edilmedi

```
param              value  lang  dataset                  wer       Δwer_pp  base
loudnorm_linear    true   en    LibriSpeech test-clean   0.01946   0.000    0.01946
loudnorm_linear    true   en    LibriSpeech test-clean   0.02087   0.000    0.02087
loudnorm_linear    true   tr    FLEURS tr_tr test        0.06248   0.000    0.06248
loudnorm_linear    true   tr    FLEURS tr_tr test        0.07081   0.000    0.07081
loudnorm_linear    true   tr    MediaSpeech TR           0.09956   0.000    0.09956
loudnorm_linear    true   tr    MediaSpeech TR           0.10198   0.000    0.10198
```

**Yorum:** Flip `linear=true` vs default `false`, **sadece** `lufs_target=-16`
merkezinde yapildi. Her 6 kosuda Δwer_pp = 0.000. ANCAK ayni log dosyasinda
`lufs_target=-20 + linear=false + MediaSpeech TR` kombinasyonu **+89 pp
felaket** uretti (asagi bak). Dynamic loudnorm modu uç hedeflerde patliyor;
`linear=true` o trial'i muhtemelen kurtarabilirdi ama test etmedik.

**Karar:** Parametre sweep'te tutuluyor. IAMS L1 screening muhtemelen
insensitive diye isaretleyecek, L5 ablation gercek toksik/kurtarici
noktalari eleyecek. Ek trial maliyeti: L2 exhaustive'de **2 trial**.

### 2) `alimiter release_ms` — zaten hardcoded, kanit onayladi

```
param              value  lang  dataset                  wer       Δwer_pp  base
limiter_release    150    en    LibriSpeech test-clean   0.01946   0.000    0.01946
limiter_release    150    en    LibriSpeech test-clean   0.02087   0.000    0.02087
limiter_release    150    tr    FLEURS tr_tr test        0.06248   0.000    0.06248
limiter_release    150    tr    FLEURS tr_tr test        0.07081   0.000    0.07081
limiter_release    150    tr    MediaSpeech TR           0.09956   0.000    0.09956
limiter_release    150    tr    MediaSpeech TR           0.10198   0.000    0.10198
```

**Yorum:** alimiter `release` parametresi zaten asrbench parametre uzayinda
tunable degildi; `ffmpeg_pipeline.py` satir ~124'de `attack=5:release=50`
olarak hardcoded. SafeScribeAI phase1 bu varsayimi 50 ms vs 150 ms
karsilastirmasiyla dogruladi; 6 kosuda sinyal yok. Kod icinde neden
hardcoded tuttugumuzu hatirlamak icin yorum eklendi.

**Caveat:** Sadece `limit=0.99` tavaninda (≈-0.087 dBFS) test edildi. Daha
dusuk tavanlarda (-3 dBFS) release hareketi farkli olabilir — ancak sweep
yuzeyine eklemek icin once bu parity kanitlanmali.

## Hardcoded (Tunable Ama Bilerek Sabit) Parametreler

Asagidaki parametreler WER'i etkiliyor (test ettigimizde 0 degil), ama
ya tuning maliyeti faydasindan yuksek, ya da tipik dagitimlarda default
yeterli — bu yuzden sweep yuzeyine alinmadi. Buraya ekleme yapmadan once
pristine-baseline kanit taleps edin.

| Parametre | Default | max\|Δwer_pp\| | avg\|Δwer_pp\| | Not |
|---|---|---|---|---|
| alimiter `attack_ms` | `5` | 2.846 | 0.768 | Aktif. Tunable yapmak icin pristine baseline'dan yeniden olcum gerekir. |
| loudnorm `tp` (true-peak) | `-1.5` | 2.388 | 0.695 | EBU R128 streaming default; dokunmadik. |

Bu ikisi icin kanit kayitlari ayni log dosyasinda: `limiter_attack=2` /
`limiter_attack=20` ve `loudnorm_tp=-0.5` / `loudnorm_tp=-3.0` satirlarina
bakin.

## Gozlenen Toksik Kombinasyon (Hariç Tutulmadi — Sadece Belge)

`preprocess.lufs_target=-20` + `loudnorm_linear=false` + MediaSpeech TR
(gurultulu TR medya yayinlari) kombinasyonu **Δwer_pp = +89.2** uretti —
transkript neredeyse tamamen bozuldu. Bu bir **toksik etkilesim**,
"isslevsiz" degil, bu yuzden parametre uzayindan cikartilmadi. IAMS L5
ablation katmani otomatik eleyecek.

```
loudnorm_i  -20  tr  MediaSpeech TR  wer=0.99164  Δwer_pp=89.208  base=0.09956
loudnorm_i  -20  tr  MediaSpeech TR  wer=0.99204  Δwer_pp=89.006  base=0.10198
```

Bu kayit burada tutuluyor cunku:

1. Gelecekte `lufs_target` araligini daraltmak istersek elimizde kanit
   bulunmali.
2. `loudnorm_linear=true` bu toksik kombinasyonu kurtarabilir mi? — test
   edilmedi. (bkz. `loudnorm_linear` RESTORED gerekce)
3. Bu toksikligin ffmpeg two-pass loudnorm extreme-target patlamasi mi,
   yoksa downstream bir bug mi oldugu henuz dogrulanmadi.

**Range daraltma karari:** Kullanici mevcut `[-24, -10]` araligini korumayi
sectigi icin daraltilmadi. IAMS L5 ablation toksik noktayi otomatik
prune edecek.

## Clean-Corpus Skip Kurali (2026-04-15)

LibriSpeech test-clean, TED-LIUM clean, VoxForge gibi studio-quality
audiobook korpuslari preprocessing-insensitive olarak kanitlandi. Bunlar
IAMS matrix'inde **yalnizca `space_clean.yaml` ile** eslenmeli; asla
`space_noisy.yaml` ile calistirilmamali.

### Kanit

SafeScribeAI phase1 sweep, 40 preprocessing varyasyonu × 3 dataset:

| Dataset | base WER% | max\|Δ\| | noise SE* | signal? |
|---|---|---|---|---|
| **LibriSpeech test-clean EN** | 1.95 | 0.281 pp | 0.291 pp | **NO** |
| FLEURS tr_tr | 6.25 | 3.887 pp | 0.510 pp | YES |
| MediaSpeech TR (noisy) | 9.96 | 89.249 pp | 0.631 pp | YES |

*noise SE formulu: `sqrt(WER × (1-WER) / N_words)`, 15 dk × 150 wpm
≈ 2250 kelime. Binomial WER gurultu zemini.

**Yorum:** LibriSpeech test-clean uzerinde olculen TUM preprocessing
deltalari, binomial noise floor'a esit veya daha kucuk. Yani bu
korpusta preprocessing degisiklikleri istatistiksel olarak **hic**
tespit edilemiyor — ne pozitif ne negatif sinyal var, sadece gurultu.

Noisy datasetler (FLEURS TR, MediaSpeech TR) net sinyal veriyor (max
delta SE'den 7-140× buyuk). Bu yuzden noisy sweep space'ini SADECE
onlarla eslesmek anlamli.

### Neden Dile Bagli Degil?

- LibriSpeech test-clean: audiobook studio kayitlari, tek konusmacili,
  yakin-ideal SNR → restoration preproc sadece minimal bozulma eklee
- Ayni mantik herhangi bir clean audiobook korpusu icin gecerli
  (TED-LIUM, VoxForge, LibriSpeech dev-clean, LibriSpeech-R)
- Dil farketmez: hipotetik "LibriSpeech DE" de ayni davranirdi.
- Belirleyici **kaynak ses kalitesi**, dil degil.

Bu yuzden kural **corpus-quality-based** olarak ifade edildi, dil-bazli
olarak degil.

### Nasil Uygulanir?

1. **`optimize_matrix.json`** icinde clean corpora (LibriSpeech, TED-LIUM,
   vb.) icin sadece `space_clean.yaml` ile entry acin. `space_noisy.yaml`
   ile ikinci bir entry **acmayin**. 2026-04-15'te `noisy-en-librispeech-seq`
   ve `noisy-en-librispeech-batch5` bu kural ile kaldirildi.
2. **`spaces/space_clean.yaml`** header'inda ayni kural yaziyor — okuyucular
   YAML duzenlerken bu uyariyi gorsun.
3. **Noisy sweep icin:** EN tarafinda `earnings22` (konferans, gurultulu),
   TR tarafinda `mediaspeech` (medya yayinlari, gurultulu) dataset'lerini
   kullanin — asrbench `dataset_manager.py` ikisi de destekliyor.

### Caveat

Bu olcum SafeScribeAI'nin tuned baseline'ina gore yapildi (bkz. yukaridaki
Baseline Metodoloji Uyarisi). Pristine baseline ile yeniden olculurse
deltalar biraz buyuyebilir ama noise-floor argumani dayaniklidir: ~%2 WER
× 2250 kelime her durumda ~0.3 pp SE verir, yani preprocessing efektleri
bu esigin ALTINDA kalmaya devam eder. Baseline degisimine duyarsiz bir
istatistiksel alt sinir.

## Test Edilmeyen Ama Extreme-Tail Adayi Parametreler

SafeScribeAI phase1 sadece 10 parametreyi test etti. Asagidaki asrbench
parametreleri hic test edilmedi ve range narrowing icin kanit yok:

- `beam_size`, `temperature`, `patience`, `repetition_penalty`,
  `no_repeat_ngram_size`, `length_penalty`
- `compression_ratio_threshold`, `log_prob_threshold`, `no_speech_threshold`,
  `hallucination_silence_threshold`
- `condition_on_previous_text`, `prompt_reset_on_temperature`, `chunk_length`,
  `max_new_tokens`
- VAD altparametreleri: `vad_threshold`, `vad_min_speech_duration_ms`,
  `vad_max_speech_duration_s`, `vad_min_silence_duration_ms`,
  `vad_speech_pad_ms`
- `preprocess.format`, `preprocess.sample_rate`, `preprocess.drc_ratio`,
  `preprocess.noise_reduce`, `preprocess.notch_hz`, `preprocess.lowpass_hz`,
  `preprocess.limiter_ceiling_db`, `preprocess.preemph_coef`, vb.

Bunlarin range'lerini daraltmak icin yeni bir pristine-baseline sweep
calistirilmalidir.

## Referanslar

- `SafeScribeAI/scripts/wer_results_log.jsonl` — kaynak JSONL (pretty-printed
  kayitlar, bos satirla ayrilmis).
- `asrbench/data/spaces/faster_whisper_full.yaml` — loudnorm_linear RESTORED
  bloku.
- `spaces/space_noisy.yaml` — loudnorm_linear RESTORED bloku.
- `asrbench/preprocessing/ffmpeg_pipeline.py` — `loudnorm tp` ve `alimiter
  attack/release` hardwiring gerekcesi.
- `tests/unit/test_space_yaml.py` — parametre sayisi / liste assertion'lari
  (26 params, 16 preprocess).

## Guncelleme Kurallari

Bu dokumanin degismesi gereken durum: optimizer sweep yuzeyinden yeni
bir parametre cikardiniz veya hardcoded yaptiniz.

- **Parametre cikarirken** yeni bir satir ekleyin: (1) ozet tabloya ekleyin,
  (2) kanit tablosu bloku olarak orijinal satirlari yapistirin, (3) ilgili
  YAML/py dosyasina kisa EXCLUDED yorumu ekleyin ki okuyan burayi bulsun.
- **Pristine baseline gerekli:** Yeni bir exclusion icin SafeScribeAI'nin
  tuned baseline'i YETERLI DEGIL. Iki baseline rejiminde de sinyal olmayan
  parametreler icin exclusion serbest; sadece birinde olan parametreler
  sweep'te tutulmali.
- **Eski kaydi silmeyin**; hariç tutma kararini geri almak gerekirse
  tarihsel kanit lazim olur.
