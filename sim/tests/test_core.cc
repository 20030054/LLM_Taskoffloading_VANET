// L0 unit tests for the pure-C++ decision core. Build & run: make -C sim/tests
#include "../src/core/Core.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

using namespace liteoff;

static int failures = 0, checks = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        ++checks;                                                                \
        if (!(cond)) {                                                           \
            ++failures;                                                          \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
        }                                                                        \
    } while (0)
#define NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) <= (tol))

static Context makeCtx()
{
    Context ctx;
    ctx.now = 10.0;
    ctx.task.id = 1;
    ctx.task.created = 10.0;
    ctx.task.tmax = 0.3;
    ctx.task.bytes = 100e3;
    ctx.task.cycles = 1e9;
    ctx.task.resultBytes = 1e3;
    ctx.task.cls = TaskClass::Perception;
    ctx.self.cpuHz = 2e9;  // local: 0.5 s > 0.3 s budget
    Candidate rsu;
    rsu.nodeId = 100;
    rsu.kind = NodeKind::Rsu;
    rsu.dist = 120;
    rsu.linkLifetime = 15;
    rsu.rateBps = 12e6;
    rsu.cpuHz = 20e9;
    rsu.trust = 0.95;
    rsu.slot = 1;
    Candidate veh;
    veh.nodeId = 7;
    veh.kind = NodeKind::Vehicle;
    veh.dist = 60;
    veh.linkLifetime = 0.2;  // leaves range before result returns
    veh.rateBps = 18e6;
    veh.cpuHz = 3e9;
    veh.slot = 2;
    ctx.cands = {rsu, veh};
    return ctx;
}

static void testLinkLifetime()
{
    // same velocity -> never separates
    NEAR(linkLifetime({0, 0}, {10, 0}, {100, 0}, {10, 0}, 300), 60, 1e-9);
    // receding at 10 m/s from 100 m, range 300 -> 20 s
    NEAR(linkLifetime({0, 0}, {0, 0}, {100, 0}, {10, 0}, 300), 20, 1e-6);
    // approaching from -100 at +10 m/s -> passes and leaves at +300: 40 s
    NEAR(linkLifetime({0, 0}, {0, 0}, {-100, 0}, {10, 0}, 300), 40, 1e-6);
    // already out of range
    NEAR(linkLifetime({0, 0}, {0, 0}, {400, 0}, {0, 0}, 300), 0, 1e-9);
}

static void testEstimation()
{
    EstParams p;
    Context ctx = makeCtx();
    NEAR(estimateDecision(ctx, 0, 0, p), 0.5, 1e-9);
    double r = estimateDecision(ctx, 1, 0, p);
    // upload 100kB at 12M*0.6 = 0.1111 s, exec 0.05, down 1kB 0.0011
    NEAR(r, 100e3 * 8 / 7.2e6 + 0.05 + 1e3 * 8 / 7.2e6, 1e-9);
    CHECK(std::isinf(estimateDecision(ctx, 2, 0, p)));  // link too short
    CHECK(std::isinf(estimateDecision(ctx, 3, 0, p)));  // not a slot
    // split: keep 50 % local
    ctx.task.splittable = true;
    double s = estimateDecision(ctx, 1, 50, p);
    NEAR(s, std::max(0.25, 50e3 * 8 / 7.2e6 + 0.025 + 500 * 8 / 7.2e6), 1e-9);
    CHECK(rateForMcs(mcsForDistance(50, true)) == 18e6);
    CHECK(rateForMcs(mcsForDistance(50, false)) == 6e6);
}

static void testRuleOracle()
{
    EstParams p;
    RuleWeights w;
    Context ctx = makeCtx();
    Decision d = rulePolicy(ctx, w, p);
    CHECK(d.slot == 1);  // RSU is the only feasible target
    Decision o = oraclePolicy(ctx, p, true);
    CHECK(o.slot == 1);
    // if local is fast, rule keeps it local
    ctx.self.cpuHz = 10e9;
    CHECK(rulePolicy(ctx, w, p).slot == 0);
    // nothing feasible -> min estimate
    Context c2 = makeCtx();
    c2.task.tmax = 0.01;
    CHECK(rulePolicy(c2, w, p).slot == 1);
}

static void testFilter()
{
    EstParams p;
    RuleWeights w;
    Context ctx = makeCtx();
    std::vector<Candidate> all = ctx.cands;
    for (int i = 0; i < 8; ++i) {
        Candidate c;
        c.nodeId = 200 + i;
        c.kind = NodeKind::Vehicle;
        c.linkLifetime = 5 + i;
        c.rateBps = 6e6;
        c.cpuHz = 2e9;
        c.dist = 50.0 * i;
        all.push_back(c);
    }
    auto f = filterCandidates(all, ctx.task, 0.1, 5, w, p);
    CHECK(f.size() == 5);
    for (size_t i = 0; i < f.size(); ++i) CHECK(f[i].slot == (int) i + 1);
    for (auto& c : f) CHECK(c.nodeId != 7);  // short-lived link removed
    for (size_t i = 1; i < f.size(); ++i) CHECK(f[i - 1].score >= f[i].score);
}

static void testParserValidator()
{
    auto a = parseAnswer("target=2; split=30; conf=0.8");
    CHECK(a.ok && a.slot == 2 && a.split == 30 && std::fabs(a.conf - 0.8) < 1e-9);
    CHECK(parseAnswer("Target: c1, split: 0").slot == 1);
    CHECK(parseAnswer("target=local").slot == 0);
    CHECK(parseAnswer("TARGET = 0 ; split = no").ok);
    CHECK(!parseAnswer("I think candidate 2 is best").ok);
    CHECK(!parseAnswer("target=two").ok);
    CHECK(!parseAnswer("target=1; split=150").ok);
    CHECK(!parseAnswer("").ok);

    EstParams p;
    Context ctx = makeCtx();
    CHECK(validate(parseAnswer("target=1; split=0"), ctx, p, 0.5) == ValidationResult::Ok);
    CHECK(validate(parseAnswer("garbage"), ctx, p, 0.5) == ValidationResult::Malformed);
    CHECK(validate(parseAnswer("target=9"), ctx, p, 0.5) == ValidationResult::NotCandidate);
    CHECK(validate(parseAnswer("target=2"), ctx, p, 0.5) == ValidationResult::Infeasible);
    CHECK(validate(parseAnswer("target=0"), ctx, p, 0.5) == ValidationResult::Infeasible);
    auto low = parseAnswer("target=1");
    low.conf = 0.2;
    CHECK(validate(low, ctx, p, 0.5) == ValidationResult::LowConfidence);
}

static void testGateCache()
{
    Context ctx = makeCtx();
    GateParams g;
    CHECK(gate(ctx, g, true, false) == GatePath::Local);
    CHECK(gate(ctx, g, false, false) == GatePath::Llm);
    CHECK(gate(ctx, g, false, true) == GatePath::Cache);
    Context one = ctx;
    one.cands.resize(1);
    CHECK(gate(one, g, false, false) == GatePath::Rule);
    g.enabled = false;
    CHECK(gate(ctx, g, true, true) == GatePath::Llm);

    DecisionCache cache(2.0);
    CHECK(!cache.lookup(ctx, 10.0).has_value());
    Decision d;
    d.slot = 1;
    cache.insert(ctx, d, 10.0);
    auto hit = cache.lookup(ctx, 11.0);
    CHECK(hit.has_value() && hit->slot == 1 && hit->path == "cache");
    CHECK(!cache.lookup(ctx, 12.5).has_value());  // expired
    // remap: same node in a different slot
    cache.insert(ctx, d, 20.0);
    Context swapped = ctx;
    swapped.now = 20.0;
    swapped.task.created = 20.0;
    std::swap(swapped.cands[0], swapped.cands[1]);
    swapped.cands[0].slot = 1;
    swapped.cands[1].slot = 2;
    CHECK(DecisionCache::key(swapped) != DecisionCache::key(ctx));
    CHECK(cache.hits == 1);
}

static void testLatencyJson()
{
    const char* path = "/tmp/liteoff_latency_test.csv";
    {
        std::ofstream f(path);
        f << "# comment\nmodel,a_ms,b_ms_per_in,c_ms_per_out,power_w,mem_mb,prefix_tok\nstudent,10,0.1,20,15,400,0\nwithprefix,10,0.1,20,15,400,50\n";
    }
    LatencyModel lm;
    CHECK(lm.load(path));
    CHECK(lm.has("student"));
    NEAR(lm.latency("student", 200, 5, 1.0), (10 + 20 + 100) / 1000.0, 1e-12);
    NEAR(lm.latency("student", 200, 5, 2.0), (10 + 20 + 100) / 2000.0, 1e-12);
    NEAR(lm.latency("withprefix", 200, 5, 1.0), (10 + 15 + 100) / 1000.0, 1e-12);
    NEAR(lm.latency("withprefix", 30, 0, 1.0), 10 / 1000.0, 1e-12);
    bool threw = false;
    try {
        lm.latency("nope", 1, 1);
    }
    catch (...) {
        threw = true;
    }
    CHECK(threw);

    bool ok = false;
    auto m = parseFlatJson(R"({"target": 2, "raw":"target=2; split=0\n", "ok":true, "x":{"a":[1,2]}, "conf":0.75})", &ok);
    CHECK(ok);
    CHECK(m["target"] == "2" && m["ok"] == "true" && m["conf"] == "0.75");
    CHECK(m["raw"] == "target=2; split=0\n");
    CHECK(m["x"] == R"({"a":[1,2]})");
    CHECK(jsonEscape("a\"b\n") == "a\\\"b\\n");
    Context ctx = makeCtx();
    std::string js = contextToJson(ctx, "all", 3);
    bool ok2 = false;
    auto top = parseFlatJson(js, &ok2);
    CHECK(ok2 && top.count("cands") && top.count("task") && top["vid"] == "3");
    NEAR(jainIndex({1, 1, 1, 1}), 1.0, 1e-12);
    NEAR(jainIndex({1, 0, 0, 0}), 0.25, 1e-12);
}

int main()
{
    std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"linkLifetime", testLinkLifetime}, {"estimation", testEstimation}, {"rule/oracle", testRuleOracle},
        {"filter", testFilter}, {"parser/validator", testParserValidator}, {"gate/cache", testGateCache},
        {"latency/json", testLatencyJson}};
    for (auto& t : tests) {
        int before = failures;
        t.second();
        std::printf("[%s] %s\n", failures == before ? " ok " : "FAIL", t.first);
    }
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
