// LiteOff-LLM decision core: pure C++17, no OMNeT++ dependency (unit-tested in sim/tests).
#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace liteoff {

enum class NodeKind { Local = 0, Vehicle = 1, Rsu = 2, Cloud = 3 };
enum class TaskClass { Safety = 0, Perception = 1, Infotainment = 2 };

const char* kindName(NodeKind k);
const char* className(TaskClass c);

struct Vec2 {
    double x = 0, y = 0;
};

struct Task {
    long id = 0;
    int origin = -1;
    TaskClass cls = TaskClass::Safety;
    double bytes = 0;        // input size [B]
    double cycles = 0;       // CPU cycles
    double resultBytes = 0;  // output size [B]
    double created = 0;      // [s]
    double tmax = 0;         // relative deadline [s]
    int prio = 0;            // 2 = high (safety), 1 = medium, 0 = low
    bool splittable = false;
    double deadline() const { return created + tmax; }
};

// One possible execution target as seen by the deciding vehicle (beacon based, possibly stale).
struct Candidate {
    int nodeId = -1;  // L2 address of the node to send to (RSU address for cloud)
    NodeKind kind = NodeKind::Vehicle;
    double dist = 0;          // [m]
    double relSpeed = 0;      // |v_self - v_cand| [m/s]
    double closing = 0;       // d(dist)/dt [m/s], negative = approaching
    double linkLifetime = 0;  // predicted time until out of range [s]
    double rateBps = 6e6;     // expected PHY rate
    double cpuHz = 1e9;
    double backlogSec = 0;  // queued work [s]
    double load = 0;        // utilisation 0..1
    int queueLen = 0;
    double loss = 0;  // recent beacon loss 0..1
    double rssi = -90;
    double trust = 0.8;
    int timeouts = 0;
    bool routeOverlap = false;
    double beaconAge = 0;
    double energy = 1;  // battery level 0..1
    double backhaulDelay = 0;  // one way, cloud only [s]
    double predDist2s = 0;     // predicted distance in 2 s (B3 digital-twin feature)
    // filled by the pipeline
    double estCompletion = 0;  // [s] from decision time
    double score = 0;
    int slot = 0;  // 1..K after filtering
};

struct SelfState {
    double cpuHz = 2e9;
    double backlogSec = 0;
    int queueLen = 0;
    double load = 0;
    double energy = 1;
    double speed = 0;
    double cbr = 0;  // channel busy ratio 0..1
    int neighbors = 0;
    double density = 0;       // veh/km^2
    double junctionIn = -1;   // s, -1 unknown
    double rsuDist = -1;      // m, -1 none
    double coverageLeft = 0;  // s until leaving best RSU coverage
    std::string roadType = "urban";
};

struct Context {
    Task task;
    SelfState self;
    std::vector<Candidate> cands;  // slots 1..K (index i -> slot i+1)
    double now = 0;
    double budget() const { return task.deadline() - now; }
};

struct Decision {
    int slot = 0;     // 0 = local, 1..K = cands[slot-1]
    int split = 0;    // percent of the task offloaded (0 = all remote when slot>0 and split==0 means 100)
    double conf = 1;  // 0..1
    std::string path = "rule";  // local|rule|cache|llm|rsu-llm|rl|fixed|oracle|fallback
    bool fromFallback = false;
};

// ---------------------------------------------------------------- estimation
struct EstParams {
    double macEfficiency = 0.6;  // fraction of PHY rate achieved by the app (headers, backoff, ACK)
    double linkMargin = 1.2;     // vehicle link lifetime must exceed est * margin
    double commRange = 300;      // [m]
};

// Expected completion time [s] of executing fraction f of task on candidate c.
double estimateRemote(const Task& t, const Candidate& c, double f, double cbr, const EstParams& p);
double estimateLocal(const Task& t, const SelfState& s, double f);
// Completion of a (possibly split) decision; returns +inf if infeasible because of link lifetime.
double estimateDecision(const Context& ctx, int slot, int splitPct, const EstParams& p);

// Time until two nodes moving linearly separate beyond range R (cap maxT). 0 if already out.
double linkLifetime(Vec2 p1, Vec2 v1, Vec2 p2, Vec2 v2, double R, double maxT = 60);

// PHY rate chosen by simple distance-based link adaptation (802.11p 10 MHz rates).
int mcsForDistance(double dist, bool adaptive);
// SNR-based link adaptation (rssi of the last beacon, noise floor), 3 dB fading margin included.
int mcsForSnr(double snrDb, bool adaptive);
double rateForMcs(int mcs);

// ---------------------------------------------------------------- rule policy
struct RuleWeights {
    double wLink = 1.0, wLoad = 1.0, wDist = 0.5, wTrust = 0.5;
    double localMargin = 0.7;  // execute locally if local est <= margin * budget
};
double ruleScore(const Candidate& c, const RuleWeights& w, double range);
// Greedy rule (= fallback). Picks local if it meets the deadline with margin, else the feasible
// candidate with best score; if nothing is feasible, the option with minimum estimated completion.
Decision rulePolicy(const Context& ctx, const RuleWeights& w, const EstParams& p);
// "basic" rule using only speed and distance (E7 / ablation)
Decision basicRulePolicy(const Context& ctx, const EstParams& p);

// Keep candidates whose link outlives the minimum transfer+execution time; top K by rule score.
std::vector<Candidate> filterCandidates(std::vector<Candidate> all, const Task& t, double cbr, int K,
                                        const RuleWeights& w, const EstParams& p);

// Full-information oracle over the given (true-state) candidate list: keep the task local when it
// meets the deadline with margin (no network risk), otherwise minimum expected completion time
// over all targets and split ratios.
Decision oraclePolicy(const Context& trueCtx, const EstParams& p, bool trySplits, double localMargin = 0.8);

// ---------------------------------------------------------------- validator
struct ParsedAnswer {
    bool ok = false;
    int slot = -1;
    int split = 0;
    double conf = -1;  // -1 = not given
    std::string error;
};
// Parses "target=<id>; split=<0-100>[; conf=<0-1>]". id: 0..K, "local", "c<k>".
ParsedAnswer parseAnswer(const std::string& raw);

enum class ValidationResult { Ok, Malformed, NotCandidate, Infeasible, LowConfidence };
const char* validationName(ValidationResult v);
ValidationResult validate(const ParsedAnswer& a, const Context& ctx, const EstParams& p, double confThreshold,
                          double slack = 1.0);

// ---------------------------------------------------------------- gate & cache
enum class GatePath { Local, Rule, Cache, Llm };
const char* gatePathName(GatePath g);
struct GateParams {
    bool enabled = true;
    double tinyCycles = 5e7;  // tasks below this go to the rule
    bool useCache = true;
};

class DecisionCache {
public:
    explicit DecisionCache(double ttl = 2.0, size_t maxEntries = 256) : ttl_(ttl), max_(maxEntries) {}
    static std::string key(const Context& ctx);
    // returns decision with slot remapped to current candidate list, if valid and not expired
    std::optional<Decision> lookup(const Context& ctx, double now);
    void insert(const Context& ctx, const Decision& d, double now);
    size_t size() const { return map_.size(); }
    long hits = 0, misses = 0;

private:
    struct Entry {
        int nodeId;
        NodeKind kind;
        int split;
        double expiry;
    };
    double ttl_;
    size_t max_;
    std::unordered_map<std::string, Entry> map_;
};

GatePath gate(const Context& ctx, const GateParams& g, bool localFeasible, bool cacheHit);

// ---------------------------------------------------------------- LLM latency model
struct LatencyEntry {
    double aMs = 0, bMsPerIn = 0, cMsPerOut = 0;
    double powerW = 10;
    double memMb = 0;
    int prefixTok = 0;  // prompt tokens that stay KV-cached between decisions (not charged)
};
class LatencyModel {
public:
    bool load(const std::string& csvPath, std::string* err = nullptr);
    void set(const std::string& model, const LatencyEntry& e) { m_[model] = e; }
    bool has(const std::string& model) const { return m_.count(model) > 0; }
    // seconds
    double latency(const std::string& model, int nIn, int nOut, double speedup = 1.0) const;
    const LatencyEntry& entry(const std::string& model) const;

private:
    std::map<std::string, LatencyEntry> m_;
};

// ---------------------------------------------------------------- misc
double jainIndex(const std::vector<double>& x);

// Minimal flat-JSON helpers (sidecar protocol). Values: number | string | bool | null.
std::string jsonEscape(const std::string& s);
std::map<std::string, std::string> parseFlatJson(const std::string& line, bool* ok = nullptr);
std::string contextToJson(const Context& ctx, const std::string& families, int vid);

}  // namespace liteoff
