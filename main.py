"""递归计算认知系统 - 主入口。

运行：
  python main.py              # 跑测试 + 两个实验
  python main.py --tests      # 仅测试
  python main.py --abcd       # 仅四组对照实验
  python main.py --compress   # 仅压缩/学习曲线实验
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_tests():
    import tests.test_core as tc
    tc.run_all()


def run_abcd():
    from experiments.run_abcd import run_all, print_comparison
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    results = run_all(seed=7, out_dir=out)
    print_comparison(results)


def run_abcd_multiseed():
    from experiments.run_abcd import run_multi_seed
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    seeds = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    agg = run_multi_seed(seeds, out_dir=out)
    print("\n" + "=" * 88)
    print("多 seed 聚合结果 (10 seeds)")
    print("=" * 88)
    for label in ["A", "B", "C", "D"]:
        m = agg[label]
        print(f"\n[{label}]")
        for k in ["valid_correct", "valid_wrong", "error_rate", "total_cost",
                  "cache_saved_cost", "yield_per_cost"]:
            print(f"  {k}: {m[k]['mean']} ± {m[k]['std']}")
        er = m["error_rate_pooled"]
        print(f"  错误率(合并): wrong={er['wrong']}/{er['n']}, "
              f"95%CI=[{er['ci_95'][0]}, {er['ci_95'][1]}]")


def run_compression():
    from experiments.run_compression import main as comp_main
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    comp_main(out_dir=out)


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if not args or "--tests" in args:
        run_tests()
    if not args or "--abcd" in args:
        run_abcd()
    if "--multiseed" in args:
        run_abcd_multiseed()
    if not args or "--compress" in args:
        run_compression()
