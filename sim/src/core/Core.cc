#include "Core.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace liteoff {

static const double INF = std::numeric_limits<double>::infinity();

const char* kindName(NodeKind k)
{
    switch (k) {
    case NodeKind::Local: return "local";
    case NodeKind::Vehicle: return "veh";
    case NodeKind::Rsu: return "rsu";
    case NodeKind::Cloud: return "cloud";
    }
    return "?";
}

const char* className(TaskClass c)
{
    switch (c) {
    case TaskClass::Safety: return "safety";
    case TaskClass::Perception: return "perception";
    case TaskClass::Infotainment: return "infotainment";
    }
    return "?";
}

// ------------------------------------------------------------------ estimation
double estimateLocal(const Task& t, const SelfState& s, double f)
{
    return s.backlogSec + t.cycles * f / s.cpuHz;
}

static double uploadTime(const Task& t, const Candidate& c, double f, double cbr, const EstParams& p)
{
    double eff = c.rateBps * p.macEfficiency * std::max(0.1, 1.0 - cbr);
    return t.bytes * f * 8.0 / eff;
}

double estimateRemote(const Task& t, const Candidate& c, double f, double cbr, const EstParams& p)
{
    double eff = c.rateBps * p.macEfficiency * std::max(0.1, 1.0 - cbr);
    double up = t.bytes * f * 8.0 / eff;
    double down = t.resultBytes * f * 8.0 / eff;
    double bh = (c.kind == NodeKind::Cloud) ? 2.0 * c.backhaulDelay : 0.0;
    double exec = t.cycles * f / c.cpuHz;
    return up + bh + c.backlogSec + exec + down;
}

static bool linkFeasible(const Task& t, const Candidate& c, double f, double est, double cbr, const EstParams& p)
{
    if (c.kind == NodeKind::Vehicle) return c.linkLifetime >= est * p.linkMargin;
    // infrastructure: upload must finish inside coverage, result can be relayed over the backhaul
    return c.linkLifetime >= uploadTime(t, c, f, cbr, p) * p.linkMargin;
}

double estimateDecision(const Context& ctx, int slot, int splitPct, const EstParams& p)
{
    if (slot <= 0) return estimateLocal(ctx.task, ctx.self, 1.0);
    if (slot > (int) ctx.cands.size()) return INF;
    const Candidate& c = ctx.cands[slot - 1];
    double keepLocal = (ctx.task.splittable ? std::clamp(splitPct, 0, 100) : 0) / 100.0;
    double f = 1.0 - keepLocal;
    if (f <= 0) return estimateLocal(ctx.task, ctx.self, 1.0);
    double r = estimateRemote(ctx.task, c, f, ctx.self.cbr, p);
    if (!linkFeasible(ctx.task, c, f, r, ctx.self.cbr, p)) return INF;
    double l = keepLocal > 0 ? estimateLocal(ctx.task, ctx.self, keepLocal) : 0.0;
    return std::max(r, l);
}

double linkLifetime(Vec2 p1, Vec2 v1, Vec2 p2, Vec2 v2, double R, double maxT)
{
    double dx = p2.x - p1.x, dy = p2.y - p1.y;
    double dvx = v2.x - v1.x, dvy = v2.y - v1.y;
    double c = dx * dx + dy * dy - R * R;
    if (c > 0) return 0.0;
    double a = dvx * dvx + dvy * dvy;
    if (a < 1e-9) return maxT;
    double b = 2 * (dx * dvx + dy * dvy);
    double disc = b * b - 4 * a * c;
    double t = (-b + std::sqrt(std::max(0.0, disc))) / (2 * a);
    return std::clamp(t, 0.0, maxT);
}

int mcsForDistance(double dist, bool adaptive)
{
    if (!adaptive) return 2;  // 6 Mbps
    if (dist <= 80) return 5;   // 18 Mbps
    if (dist <= 150) return 4;  // 12 Mbps
    if (dist <= 230) return 3;  // 9 Mbps
    return 2;                   // 6 Mbps
}

int mcsForSnr(double snrDb, bool adaptive)
{
    if (!adaptive) return 2;
    // approximate 10 MHz 802.11p sensitivity steps + 3 dB margin
    if (snrDb >= 24) return 5;  // 18 Mbps
    if (snrDb >= 20) return 4;  // 12 Mbps
    if (snrDb >= 17) return 3;  // 9 Mbps
    return 2;                   // 6 Mbps
}

double rateForMcs(int mcs)
{
    static const double r[] = {3e6, 4.5e6, 6e6, 9e6, 12e6, 18e6, 24e6, 27e6};
    if (mcs < 0 || mcs > 7) return 6e6;
    return r[mcs];
}

// ------------------------------------------------------------------ rule
double ruleScore(const Candidate& c, const RuleWeights& w, double range)
{
    double load = std::max(c.load, std::min(1.0, c.backlogSec));
    double link = std::min(c.linkLifetime, 20.0) / 20.0;
    double dist = (c.kind == NodeKind::Cloud) ? 0.5 : std::min(1.0, c.dist / std::max(1.0, range));
    return w.wLink * link - w.wLoad * load - w.wDist * dist + w.wTrust * c.trust;
}

Decision rulePolicy(const Context& ctx, const RuleWeights& w, const EstParams& p)
{
    Decision d;
    d.path = "rule";
    double budget = ctx.budget();
    double localEst = estimateLocal(ctx.task, ctx.self, 1.0);
    if (localEst <= w.localMargin * budget) {
        d.slot = 0;
        return d;
    }
    int best = -1;
    double bestScore = -INF;
    for (size_t i = 0; i < ctx.cands.size(); ++i) {
        double est = estimateDecision(ctx, (int) i + 1, 0, p);
        if (est > budget) continue;
        double s = ruleScore(ctx.cands[i], w, p.commRange);
        if (s > bestScore) {
            bestScore = s;
            best = (int) i + 1;
        }
    }
    if (best > 0) {
        d.slot = best;
        return d;
    }
    if (localEst <= budget) {
        d.slot = 0;
        return d;
    }
    double bestEst = localEst;
    d.slot = 0;
    for (size_t i = 0; i < ctx.cands.size(); ++i) {
        double est = estimateDecision(ctx, (int) i + 1, 0, p);
        if (est < bestEst) {
            bestEst = est;
            d.slot = (int) i + 1;
        }
    }
    return d;
}

Decision basicRulePolicy(const Context& ctx, const EstParams& p)
{
    Decision d;
    d.path = "rule";
    double localEst = estimateLocal(ctx.task, ctx.self, 1.0);
    if (localEst <= 0.7 * ctx.budget() || ctx.cands.empty()) return d;
    double best = -INF;
    for (size_t i = 0; i < ctx.cands.size(); ++i) {
        const auto& c = ctx.cands[i];
        double s = -(c.kind == NodeKind::Cloud ? 0.5 : c.dist / p.commRange) - c.relSpeed / 30.0;
        if (s > best) {
            best = s;
            d.slot = (int) i + 1;
        }
    }
    return d;
}

std::vector<Candidate> filterCandidates(std::vector<Candidate> all, const Task& t, double cbr, int K,
                                        const RuleWeights& w, const EstParams& p)
{
    std::vector<Candidate> keep;
    for (auto& c : all) {
        double eff = c.rateBps * p.macEfficiency * std::max(0.1, 1.0 - cbr);
        double up = t.bytes * 8.0 / eff;
        double exec = t.cycles / c.cpuHz;
        double minTime = (c.kind == NodeKind::Vehicle) ? up + exec : up;
        if (c.linkLifetime < minTime) continue;
        c.score = ruleScore(c, w, p.commRange);
        keep.push_back(c);
    }
    std::stable_sort(keep.begin(), keep.end(), [](const Candidate& a, const Candidate& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.nodeId != b.nodeId) return a.nodeId < b.nodeId;
        return (int) a.kind < (int) b.kind;
    });
    if ((int) keep.size() > K) keep.resize(K);
    for (size_t i = 0; i < keep.size(); ++i) keep[i].slot = (int) i + 1;
    return keep;
}

Decision oraclePolicy(const Context& ctx, const EstParams& p, bool trySplits, double localMargin)
{
    Decision d;
    d.path = "oracle";
    d.slot = 0;
    double best = estimateLocal(ctx.task, ctx.self, 1.0);
    if (best <= localMargin * ctx.budget()) return d;
    static const int splits[] = {0, 30, 50, 70};
    for (size_t i = 0; i < ctx.cands.size(); ++i) {
        for (int s : splits) {
            if (s > 0 && !(trySplits && ctx.task.splittable)) break;
            double est = estimateDecision(ctx, (int) i + 1, s, p);
            if (est + 1e-9 < best) {
                best = est;
                d.slot = (int) i + 1;
                d.split = s;
            }
        }
    }
    return d;
}

// ------------------------------------------------------------------ validator
static std::string lower(std::string s)
{
    for (auto& ch : s) ch = (char) std::tolower((unsigned char) ch);
    return s;
}

static bool readValue(const std::string& s, const std::string& key, std::string& out)
{
    size_t pos = 0;
    while ((pos = s.find(key, pos)) != std::string::npos) {
        size_t q = pos + key.size();
        while (q < s.size() && (s[q] == ' ' || s[q] == '"')) ++q;
        if (q < s.size() && (s[q] == '=' || s[q] == ':')) {
            ++q;
            while (q < s.size() && (s[q] == ' ' || s[q] == '"' || s[q] == '<')) ++q;
            size_t e = q;
            while (e < s.size() && (std::isalnum((unsigned char) s[e]) || s[e] == '.' || s[e] == '_')) ++e;
            out = s.substr(q, e - q);
            return !out.empty();
        }
        pos = q;
    }
    return false;
}

ParsedAnswer parseAnswer(const std::string& raw)
{
    ParsedAnswer a;
    std::string s = lower(raw);
    std::string tv;
    if (!readValue(s, "target", tv)) {
        a.error = "no target";
        return a;
    }
    if (tv == "local" || tv == "self") a.slot = 0;
    else {
        std::string num = tv;
        if (num.rfind("cand", 0) == 0) num = num.substr(4);
        else if (num[0] == 'c') num = num.substr(1);
        if (num.empty() || !std::all_of(num.begin(), num.end(), ::isdigit)) {
            a.error = "bad target '" + tv + "'";
            return a;
        }
        a.slot = std::stoi(num);
    }
    std::string sv;
    if (readValue(s, "split", sv)) {
        if (sv == "no" || sv == "none") a.split = 0;
        else {
            try {
                a.split = (int) std::lround(std::stod(sv));
            }
            catch (...) {
                a.error = "bad split";
                return a;
            }
            if (a.split < 0 || a.split > 100) {
                a.error = "split out of range";
                return a;
            }
        }
    }
    std::string cv;
    if (readValue(s, "conf", cv)) {
        try {
            a.conf = std::clamp(std::stod(cv), 0.0, 1.0);
        }
        catch (...) {
            a.conf = -1;
        }
    }
    a.ok = true;
    return a;
}

const char* validationName(ValidationResult v)
{
    switch (v) {
    case ValidationResult::Ok: return "ok";
    case ValidationResult::Malformed: return "malformed";
    case ValidationResult::NotCandidate: return "not_candidate";
    case ValidationResult::Infeasible: return "infeasible";
    case ValidationResult::LowConfidence: return "low_confidence";
    }
    return "?";
}

ValidationResult validate(const ParsedAnswer& a, const Context& ctx, const EstParams& p, double confThreshold,
                          double slack)
{
    if (!a.ok) return ValidationResult::Malformed;
    if (a.slot < 0 || a.slot > (int) ctx.cands.size()) return ValidationResult::NotCandidate;
    double budget = ctx.budget() * slack;
    double est = estimateDecision(ctx, a.slot, a.split, p);
    if (est > budget) {
        double minEst = estimateLocal(ctx.task, ctx.self, 1.0);
        for (size_t i = 0; i < ctx.cands.size(); ++i) minEst = std::min(minEst, estimateDecision(ctx, (int) i + 1, 0, p));
        if (minEst <= budget) return ValidationResult::Infeasible;
    }
    if (a.conf >= 0 && a.conf < confThreshold) return ValidationResult::LowConfidence;
    return ValidationResult::Ok;
}

// ------------------------------------------------------------------ gate & cache
const char* gatePathName(GatePath g)
{
    switch (g) {
    case GatePath::Local: return "local";
    case GatePath::Rule: return "rule";
    case GatePath::Cache: return "cache";
    case GatePath::Llm: return "llm";
    }
    return "?";
}

GatePath gate(const Context& ctx, const GateParams& g, bool localFeasible, bool cacheHit)
{
    if (!g.enabled) return GatePath::Llm;
    if (localFeasible) return GatePath::Local;
    if (ctx.cands.size() <= 1 || ctx.task.cycles < g.tinyCycles) return GatePath::Rule;
    if (g.useCache && cacheHit) return GatePath::Cache;
    return GatePath::Llm;
}

static int qbin(double v, double step, int maxBin)
{
    if (!std::isfinite(v)) return maxBin;
    return std::clamp((int) std::floor(v / step), 0, maxBin);
}

std::string DecisionCache::key(const Context& ctx)
{
    std::ostringstream k;
    k << (int) ctx.task.cls << '|' << qbin(std::log2(std::max(1.0, ctx.task.bytes / 1024.0)), 1.0, 20) << '|'
      << qbin(ctx.budget(), 0.05, 40) << '|' << qbin(ctx.self.backlogSec, 0.05, 20);
    for (const auto& c : ctx.cands) {
        k << '|' << (int) c.kind << ':' << qbin(c.linkLifetime, 2.0, 10) << ':' << qbin(std::max(c.load, std::min(1.0, c.backlogSec)), 0.2, 5)
          << ':' << qbin(c.dist, 75.0, 8);
    }
    return k.str();
}

std::optional<Decision> DecisionCache::lookup(const Context& ctx, double now)
{
    auto it = map_.find(key(ctx));
    if (it == map_.end()) {
        ++misses;
        return std::nullopt;
    }
    if (it->second.expiry < now) {
        map_.erase(it);
        ++misses;
        return std::nullopt;
    }
    Decision d;
    d.path = "cache";
    d.split = it->second.split;
    if (it->second.kind == NodeKind::Local) {
        d.slot = 0;
        ++hits;
        return d;
    }
    for (const auto& c : ctx.cands) {
        if (c.nodeId == it->second.nodeId && c.kind == it->second.kind) {
            d.slot = c.slot;
            ++hits;
            return d;
        }
    }
    ++misses;
    return std::nullopt;
}

void DecisionCache::insert(const Context& ctx, const Decision& d, double now)
{
    if (map_.size() >= max_) {
        for (auto it = map_.begin(); it != map_.end();) {
            if (it->second.expiry < now) it = map_.erase(it);
            else ++it;
        }
        if (map_.size() >= max_) map_.erase(map_.begin());
    }
    Entry e;
    e.split = d.split;
    double life = ttl_;
    if (d.slot > 0 && d.slot <= (int) ctx.cands.size()) {
        const auto& c = ctx.cands[d.slot - 1];
        e.nodeId = c.nodeId;
        e.kind = c.kind;
        life = std::min(ttl_, c.linkLifetime);
    }
    else {
        e.nodeId = -1;
        e.kind = NodeKind::Local;
    }
    e.expiry = now + life;
    map_[key(ctx)] = e;
}

// ------------------------------------------------------------------ latency model
bool LatencyModel::load(const std::string& path, std::string* err)
{
    std::ifstream in(path);
    if (!in) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    std::string line;
    bool header = true;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (header) {
            header = false;
            if (line.rfind("model", 0) == 0) continue;
        }
        std::stringstream ss(line);
        std::string name, f;
        std::vector<double> v;
        std::getline(ss, name, ',');
        while (std::getline(ss, f, ',')) v.push_back(std::stod(f));
        if (v.size() < 3) continue;
        LatencyEntry e;
        e.aMs = v[0];
        e.bMsPerIn = v[1];
        e.cMsPerOut = v[2];
        if (v.size() > 3) e.powerW = v[3];
        if (v.size() > 4) e.memMb = v[4];
        if (v.size() > 5) e.prefixTok = (int) v[5];
        m_[name] = e;
    }
    return true;
}

const LatencyEntry& LatencyModel::entry(const std::string& model) const
{
    auto it = m_.find(model);
    if (it == m_.end()) throw std::runtime_error("latency model has no entry for '" + model + "'");
    return it->second;
}

double LatencyModel::latency(const std::string& model, int nIn, int nOut, double speedup) const
{
    const auto& e = entry(model);
    double ms = e.aMs + e.bMsPerIn * std::max(0, nIn - e.prefixTok) + e.cMsPerOut * nOut;
    return std::max(0.0, ms) / 1000.0 / std::max(1e-6, speedup);
}

// ------------------------------------------------------------------ misc
double jainIndex(const std::vector<double>& x)
{
    double s = 0, s2 = 0;
    for (double v : x) {
        s += v;
        s2 += v * v;
    }
    if (x.empty() || s2 <= 0) return 1.0;
    return s * s / (x.size() * s2);
}

std::string jsonEscape(const std::string& s)
{
    std::string o;
    for (char ch : s) {
        switch (ch) {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\t': o += "\\t"; break;
        case '\r': o += "\\r"; break;
        default:
            if ((unsigned char) ch < 0x20) {
                char b[8];
                std::snprintf(b, sizeof b, "\\u%04x", ch);
                o += b;
            }
            else o += ch;
        }
    }
    return o;
}

std::map<std::string, std::string> parseFlatJson(const std::string& s, bool* ok)
{
    std::map<std::string, std::string> m;
    if (ok) *ok = false;
    size_t i = s.find('{');
    if (i == std::string::npos) return m;
    ++i;
    auto skipWs = [&]() {
        while (i < s.size() && std::isspace((unsigned char) s[i])) ++i;
    };
    auto readString = [&](std::string& out) -> bool {
        if (i >= s.size() || s[i] != '"') return false;
        ++i;
        out.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                ++i;
                switch (s[i]) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'u': i += 4; out += '?'; break;
                default: out += s[i];
                }
            }
            else out += s[i];
            ++i;
        }
        if (i >= s.size()) return false;
        ++i;
        return true;
    };
    while (true) {
        skipWs();
        if (i >= s.size()) return m;
        if (s[i] == '}') break;
        if (s[i] == ',') {
            ++i;
            continue;
        }
        std::string key, val;
        if (!readString(key)) return m;
        skipWs();
        if (i >= s.size() || s[i] != ':') return m;
        ++i;
        skipWs();
        if (i < s.size() && s[i] == '"') {
            if (!readString(val)) return m;
        }
        else if (i < s.size() && (s[i] == '{' || s[i] == '[')) {
            int depth = 0;
            size_t st = i;
            bool inStr = false;
            for (; i < s.size(); ++i) {
                char ch = s[i];
                if (inStr) {
                    if (ch == '\\') ++i;
                    else if (ch == '"') inStr = false;
                    continue;
                }
                if (ch == '"') inStr = true;
                else if (ch == '{' || ch == '[') ++depth;
                else if (ch == '}' || ch == ']') {
                    if (--depth == 0) {
                        ++i;
                        break;
                    }
                }
            }
            val = s.substr(st, i - st);
        }
        else {
            size_t st = i;
            while (i < s.size() && s[i] != ',' && s[i] != '}' && !std::isspace((unsigned char) s[i])) ++i;
            val = s.substr(st, i - st);
        }
        m[key] = val;
    }
    if (ok) *ok = true;
    return m;
}

static void num(std::ostringstream& o, const char* k, double v, bool comma = true)
{
    char b[64];
    if (!std::isfinite(v)) v = v > 0 ? 1e9 : -1e9;
    std::snprintf(b, sizeof b, "\"%s\":%.6g", k, v);
    o << b;
    if (comma) o << ',';
}

std::string contextToJson(const Context& ctx, const std::string& families, int vid)
{
    std::ostringstream o;
    const Task& t = ctx.task;
    o << '{';
    num(o, "t", ctx.now);
    o << "\"vid\":" << vid << ",\"families\":\"" << jsonEscape(families) << "\",";
    o << "\"task\":{";
    num(o, "id", (double) t.id);
    num(o, "bytes", t.bytes);
    num(o, "cycles", t.cycles);
    num(o, "result_bytes", t.resultBytes);
    num(o, "deadline_ms", t.tmax * 1000);
    num(o, "budget_ms", ctx.budget() * 1000);
    num(o, "prio", t.prio);
    o << "\"cls\":\"" << className(t.cls) << "\",\"split\":" << (t.splittable ? "true" : "false") << "},";
    const SelfState& s = ctx.self;
    o << "\"self\":{";
    num(o, "cpu_ghz", s.cpuHz / 1e9);
    num(o, "backlog_ms", s.backlogSec * 1000);
    num(o, "queue", s.queueLen);
    num(o, "load", s.load);
    num(o, "energy", s.energy);
    num(o, "speed", s.speed);
    num(o, "cbr", s.cbr);
    num(o, "local_est_ms", estimateLocal(t, s, 1.0) * 1000, false);
    o << "},\"road\":{";
    num(o, "density", s.density);
    num(o, "neighbors", s.neighbors);
    num(o, "junction_in", s.junctionIn);
    num(o, "rsu_dist", s.rsuDist);
    num(o, "coverage_left", s.coverageLeft);
    o << "\"type\":\"" << jsonEscape(s.roadType) << "\"},\"cands\":[";
    for (size_t i = 0; i < ctx.cands.size(); ++i) {
        const auto& c = ctx.cands[i];
        if (i) o << ',';
        o << "{\"slot\":" << c.slot << ",\"kind\":\"" << kindName(c.kind) << "\",\"node\":" << c.nodeId << ',';
        num(o, "dist", c.dist);
        num(o, "rel_speed", c.relSpeed);
        num(o, "closing", c.closing);
        num(o, "link_left", c.linkLifetime);
        num(o, "rate_mbps", c.rateBps / 1e6);
        num(o, "cpu_ghz", c.cpuHz / 1e9);
        num(o, "load", c.load);
        num(o, "backlog_ms", c.backlogSec * 1000);
        num(o, "queue", c.queueLen);
        num(o, "loss", c.loss);
        num(o, "rssi", c.rssi);
        num(o, "trust", c.trust);
        num(o, "timeouts", c.timeouts);
        num(o, "energy", c.energy);
        num(o, "backhaul_ms", c.backhaulDelay * 1000);
        num(o, "pred_dist_2s", c.predDist2s);
        num(o, "beacon_age", c.beaconAge);
        o << "\"route_overlap\":" << (c.routeOverlap ? "true" : "false") << '}';
    }
    o << "]}";
    return o.str();
}

}  // namespace liteoff
