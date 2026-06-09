"""Ad-hoc ephemeral-disk read benchmark for Modal.

Measures the access pattern the dataloader actually uses: random small-chunk
reads (zarr/JPEG chunks), single- and 48-thread concurrent, plus sequential.

    MODAL_ENVIRONMENT=robotics modal run egomimic/modal/bench_disk.py
"""
import modal

app = modal.App("egoverse-disk-bench")
image = modal.Image.debian_slim()


@app.function(image=image, cpu=32, ephemeral_disk=600 * 1024, timeout=900)
def bench() -> dict:
    import os, time, random
    from concurrent.futures import ThreadPoolExecutor

    D = "/tmp/disktest"
    os.makedirs(D, exist_ok=True)
    nfiles, fsize, chunk = 64, 256 * 1024 * 1024, 64 * 1024  # 16 GB total, 64 KB reads

    def drop_caches():
        try:
            os.system("sync")
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3")
            return True
        except Exception:
            return False

    # --- write 16 GB ---
    buf = os.urandom(fsize)
    t0 = time.time()
    paths = []
    for i in range(nfiles):
        p = f"{D}/f{i}.bin"
        with open(p, "wb") as f:
            f.write(buf); f.flush(); os.fsync(f.fileno())
        paths.append(p)
    write_gbps = (nfiles * fsize / 1e9) / (time.time() - t0)

    dropped = drop_caches()

    # --- sequential read ---
    t0 = time.time(); total = 0
    for p in paths:
        with open(p, "rb") as f:
            while True:
                b = f.read(8 * 1024 * 1024)
                if not b:
                    break
                total += len(b)
    seq_gbps = (total / 1e9) / (time.time() - t0)

    def rand_reads(n):
        for _ in range(n):
            p = random.choice(paths); off = random.randint(0, fsize - chunk)
            with open(p, "rb") as f:
                f.seek(off); f.read(chunk)
        return n

    # --- random 64KB, single thread (cold) ---
    drop_caches()
    t0 = time.time(); rand_reads(3000); st = time.time() - t0
    # --- random 64KB, 48 concurrent (cold) ---
    drop_caches()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=48) as ex:
        list(ex.map(rand_reads, [500] * 48))
    ct = time.time() - t0; tot48 = 48 * 500

    return {
        "drop_caches_worked": dropped,
        "write_GBps": round(write_gbps, 2),
        "seq_read_GBps": round(seq_gbps, 2),
        "rand64k_1thr_iops": round(3000 / st), "rand64k_1thr_MBps": round(3000 * chunk / 1e6 / st),
        "rand64k_48thr_iops": round(tot48 / ct), "rand64k_48thr_MBps": round(tot48 * chunk / 1e6 / ct),
    }


@app.local_entrypoint()
def main() -> None:
    print("DISK BENCH:", bench.remote())
