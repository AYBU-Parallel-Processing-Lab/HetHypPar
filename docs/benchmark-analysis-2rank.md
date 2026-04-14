# Benchmark Analizi: 2 Rank Testleri

**Tarih:** 2026-03-22
**Kapsam:** 213 matris, 4 weight profili (50_50, 66_33, 75_25, 90_10), %3–%50 imbalance, ~32,000 test

## Soru 1: İterasyon Sayısı

20 iterasyon gerçekten az. Literatürde BiCGStab için genellikle **500–2000** iterasyon kullanılıyor (matrisin condition number'ına bağlı). 1000 iterasyon makul bir değer. Şu an 213 matrisin sadece **78'i** (%37) converge ediyor 20 iterasyonda — 1000 iterasyona çıkarsak çok daha fazlası converge olur ve dataset büyür.

## Soru 2: Converge Olmadan Çalıştırma

Speedup analizi için converge olması şart değil, çünkü SpMV süresi iterasyon sayısından bağımsız (her iterasyonda aynı işlem yapılıyor). Converge şartını kaldırırsak 213 matrisin tamamını kullanabiliriz. Condition number düşürmek için preconditioning (ILU, Jacobi) de bir seçenek ama implementasyon gerektirir.

## Soru 3: Cut=0 Olan Matrisler

**213 matrisin 44'ünde** (%20.7) en az bir partition konfigürasyonunda Cut=0 çıkıyor. Toplam 15,908 partition konfigürasyonunun **453'ünde** (%2.8) Cut=0.

### Detaylar

- **Gerçek blok-diagonal** (50_50 weight'te bile Cut=0): sadece **1 matris** (`ex35`)
- **66_33'te Cut=0 olan** (kısmen blok-diagonal): **8 matris** (ex19, li, shyy161, std1_Jac2_db, std1_Jac3_db, Zd_Jac2_db, Zd_Jac3_db, Zd_Jac6_db)
- **Sadece 90_10 gibi aşırı weight'lerde Cut=0**: **35 matris** — bunlar blok-diagonal değil, PaToH yüksek weight + yüksek imbalance toleransıyla tüm satırları tek partition'a koymaya meyilli olduğu için Cut=0 çıkıyor.

### Cut=0 Dağılımı (Weight'e Göre)

| Weight | Cut=0 Sayısı |
|--------|:------------:|
| 90_10  | 311          |
| 75_25  | 88           |
| 66_33  | 51           |
| 50_50  | 3            |

### Cut=0 Dağılımı (Imbalance'a Göre)

| Imbalance (%) | Cut=0 Sayısı |
|---------------|:------------:|
| 50            | 203          |
| 20            | 131          |
| 10            | 49           |
| 5             | 40           |
| 3             | 30           |

## Soru 4: MPI+GPU vs GPU

**2 rank'lık kurulumda MPI+GPU, hiçbir matriste GPU'dan hızlı değil** (SpMV bazında 0/211).

| Metrik | Değer |
|--------|:-----:|
| En iyi durum | 1.24x yavaş |
| Ortalama | 1.89x yavaş |
| En kötü durum | 5.80x yavaş |

### Nedenler

1. Sadece 2 rank var (1 CPU + 1 GPU) — GPU zaten tüm işi tek başına yapabiliyor
2. MPI iletişim overhead'i ekleniyor (özellikle cut > 0 olan matrislerde)
3. CPU rank'ı GPU'ya yardım etmek yerine yük oluyor

MPI+GPU'nun GPU'yu geçmesi için **daha fazla rank** (örn. 8 CPU + 1 GPU, 16 CPU + 1 GPU) gerekiyor — böylece CPU rank'ları toplamda GPU'nun işinin bir kısmını üstlenip iletişim overhead'ini karşılayabilir.

> **Not:** `everything_total` bazında 31 matriste MPI+GPU daha hızlı çıkıyor, ama bu GPU'nun cuSPARSE descriptor oluşturma süresinin daha yüksek olmasından kaynaklanıyor, SpMV'den değil.

## Ek: MPI Segfault Fix'i

Benchmark sonuçlarını analiz ederken **6,468 başarısız testin 5,429'unun** (%84) aynı kök nedenden kaynaklandığı tespit edildi: `src/util/hhp_matrix.c` içindeki `internal_setup_communication()` fonksiyonunda **MPI send buffer use-after-free** race condition'ı.

### Sorun

İkinci MPI iletişim turunda global kolon indeksleri rank'lar arası gönderilirken `MPI_Isend` + `MPI_Request_free` (fire-and-forget) pattern'i kullanılıyordu. Send buffer (`recv_gJ`) sadece receive'ler bittikten sonra free ediliyordu — ama `MPI_Request_free` send'in tamamlandığını garanti etmiyor. MPI hâlâ buffer'dan okurken free edilince, alıcı rank `send.J`'de çöp veri alıyor ve `locmap` lookup'ında segfault oluyor.

### Etkilenen Matrisler

Yapısal olarak asimetrik matrisler en çok etkilendi — lopsided iletişim pattern'leri bir rank'ın diğerinden çok daha hızlı bitirmesine yol açıyor:

| Matris       | Segfault Oranı |
|-------------|:--------------:|
| g7jac080sc  | %97            |
| g7jac080    | %97            |
| lhr14       | %92            |
| bayer02     | %90            |
| g7jac100sc  | %90            |
| fd18        | %88            |

### Fix

Send request'leri track edip, buffer'ı free etmeden önce `MPI_Waitall` ile tamamlanmalarını bekliyoruz. Fix uygulandıktan sonra en kötü 4 matris üzerinde 10'ar test yapıldı — **40/40 başarılı**.

Detaylı teknik dokümantasyon: [`docs/fix-mpi-send-use-after-free.md`](fix-mpi-send-use-after-free.md)

### Kalan Hatalar

| Hata | Sayı | Açıklama |
|------|:----:|----------|
| Vector of size 0 | 876 | Bir rank'a sıfır satır atayan partition'lardan kaynaklanıyor |
| Missing matrix file | 162 | `memplus_k6.mtx` dosyası eksik |
