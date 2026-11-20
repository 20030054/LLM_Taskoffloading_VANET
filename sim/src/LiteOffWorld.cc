#include "LiteOffWorld.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <sstream>
#include <thread>

#include "LiteOffNodeApp.h"

using namespace omnetpp;

namespace liteoff {

Define_Module(LiteOffWorld);

LiteOffWorld* LiteOffWorld::instance_ = nullptr;

const char* trafficName(Traffic t)
{
    static const char* n[] = {"beacon", "task", "result", "llmctl", "model", "fed"};
    return n[(int) t];
}

// ------------------------------------------------------------------ side-car client
SidecarClient::~SidecarClient() { close(); }

void SidecarClient::close()
{
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
}

bool SidecarClient::connect(const std::string& host, int port, double timeoutSec, std::string* err)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds((long) (timeoutSec * 1000));
    while (true) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t) port);
        if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) {
            hostent* he = gethostbyname(host.c_str());
            if (!he) {
                ::close(fd);
                if (err) *err = "cannot resolve " + host;
                return false;
            }
            std::memcpy(&a.sin_addr, he->h_addr_list[0], he->h_length);
        }
        if (::connect(fd, (sockaddr*) &a, sizeof a) == 0) {
            int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            fd_ = fd;
            return true;
        }
        ::close(fd);
        if (std::chrono::steady_clock::now() > deadline) {
            if (err) *err = "cannot connect to side-car at " + host + ":" + std::to_string(port);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

std::string SidecarClient::request(const std::string& line, double* wallMs)
{
    if (fd_ < 0) return "";
    auto t0 = std::chrono::steady_clock::now();
    std::string out = line + "\n";
    size_t off = 0;
    while (off < out.size()) {
        ssize_t n = ::send(fd_, out.data() + off, out.size() - off, 0);
        if (n <= 0) {
            close();
            return "";
        }
        off += (size_t) n;
    }
    while (true) {
        size_t nl = buf_.find('\n');
        if (nl != std::string::npos) {
            std::string r = buf_.substr(0, nl);
            buf_.erase(0, nl + 1);
            if (wallMs) *wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            return r;
        }
        char tmp[65536];
        ssize_t n = ::recv(fd_, tmp, sizeof tmp, 0);
        if (n <= 0) {
            close();
            return "";
        }
        buf_.append(tmp, (size_t) n);
    }
}

// ------------------------------------------------------------------ module
LiteOffWorld* LiteOffWorld::get()
{
    if (!instance_) throw cRuntimeError("LiteOffWorld module 'world' not found in network");
    return instance_;
}

void LiteOffWorld::initialize()
{
    instance_ = this;
    warmup_ = par("warmup").doubleValue();
    scenarioTag_ = par("scenarioTag").stdstringValue();
    std::string lf = par("latencyFile").stdstringValue();
    if (!lf.empty()) {
        std::string err;
        if (!latency_.load(lf, &err)) throw cRuntimeError("latency model: %s", err.c_str());
    }
    sidecarHost_ = par("sidecarHost").stdstringValue();
    sidecarPort_ = par("sidecarPort").intValue();
    sidecarEnabled_ = sidecarPort_ > 0;
    std::string scheme = par("scheme").stdstringValue();
    bool needs = scheme == "liteoff" || scheme == "b1" || scheme == "b2" || scheme == "b3" || scheme == "b4" || scheme == "b5";
    if (needs && !sidecarEnabled_) throw cRuntimeError("scheme '%s' needs the Python side-car (set *.world.sidecarPort)", scheme.c_str());
    if (sidecarEnabled_) {
        std::string err;
        if (!sidecar_.connect(sidecarHost_, sidecarPort_, par("sidecarConnectTimeout").doubleValue(), &err)) throw cRuntimeError("%s", err.c_str());
        std::ostringstream o;
        o << "{\"op\":\"hello\",\"scheme\":\"" << scheme << "\",\"run\":\"" << jsonEscape(scenarioTag_) << "\",\"rlMode\":\""
          << par("rlMode").stdstringValue() << "\",\"rlWeights\":\"" << jsonEscape(par("rlWeights").stdstringValue()) << "\",\"seed\":" << getEnvir()->getConfigEx()->getVariable("seedset") << "}";
        std::string r = sidecar_.request(o.str());
        if (r.empty()) throw cRuntimeError("side-car did not answer hello");
    }
    if (needs) {
        // fail fast if the latency model lacks an entry this run will charge
        auto req = [this](const std::string& k) {
            if (!latency_.has(k)) throw cRuntimeError("latency model (%s) has no entry '%s'", par("latencyFile").stringValue(), k.c_str());
        };
        std::string mode = par("deploymentMode").stdstringValue();
        auto key = [this](const char* k, const char* m) { std::string v = par(k).stdstringValue(); return v.empty() ? par(m).stdstringValue() : v; };
        if (scheme == "liteoff" && mode != "rsu" && !par("ignoreLlmLatency").boolValue()) req(key("llmLatencyKey", "llmModel"));
        if ((scheme == "liteoff" && mode != "vehicle") || scheme == "b2" || scheme == "b3") req(key("rsuLlmLatencyKey", "rsuLlmModel"));
    }
    std::string tf = par("traceFile").stdstringValue();
    if (!tf.empty()) {
        trace_ = std::fopen(tf.c_str(), "w");
        if (!trace_) throw cRuntimeError("cannot open trace file %s", tf.c_str());
        std::fprintf(trace_, "task_id,origin,cls,created,tmax,bytes,cycles,path,target,split,status,delay,decision_delay,tx,queue,exec,result,oracle_agree,reoffloaded,fallback\n");
    }
    std::string df = par("datasetFile").stdstringValue();
    if (!df.empty()) {
        dataset_ = std::fopen(df.c_str(), "w");
        if (!dataset_) throw cRuntimeError("cannot open dataset file %s", df.c_str());
    }
    delayVec_.setName("taskDelay");
}

void LiteOffWorld::handleMessage(cMessage* msg) { delete msg; }

void LiteOffWorld::registerNode(int addr, LiteOffNodeApp* app) { nodes_[addr] = app; }
void LiteOffWorld::unregisterNode(int addr) { nodes_.erase(addr); }

LiteOffNodeApp* LiteOffWorld::node(int addr) const
{
    auto it = nodes_.find(addr);
    return it == nodes_.end() ? nullptr : it->second;
}

LiteOffNodeApp* LiteOffWorld::nearestRsu(double x, double y, double* dist) const
{
    LiteOffNodeApp* best = nullptr;
    double bd = 1e18;
    for (auto& kv : nodes_) {
        if (!kv.second->isRsu()) continue;
        Vec2 p = kv.second->truePos();
        double d = std::hypot(p.x - x, p.y - y);
        if (d < bd) {
            bd = d;
            best = kv.second;
        }
    }
    if (dist) *dist = bd;
    return best;
}

SidecarClient& LiteOffWorld::sidecar()
{
    if (!sidecar_.connected() && sidecarEnabled_) {
        std::string err;
        if (!sidecar_.connect(sidecarHost_, sidecarPort_, 5.0, &err)) throw cRuntimeError("side-car connection lost: %s", err.c_str());
    }
    return sidecar_;
}

long LiteOffWorld::storeQuery(const std::string& req)
{
    long id = nextQuery_++;
    queries_[id] = req;
    return id;
}

std::string LiteOffWorld::takeQuery(long id)
{
    auto it = queries_.find(id);
    if (it == queries_.end()) return "";
    std::string r = it->second;
    queries_.erase(it);
    return r;
}

// ------------------------------------------------------------------ metrics
void LiteOffWorld::countTx(Traffic t, double bytes, bool unicast)
{
    if (!counting()) return;
    bytes_[(int) t] += bytes;
    frames_[(int) t]++;
    if (unicast) unicast_[(int) t]++;
}

void LiteOffWorld::countTxFailed(Traffic t)
{
    if (counting()) failed_[(int) t]++;
}

void LiteOffWorld::countPath(const std::string& p)
{
    if (counting()) paths_[p]++;
}

void LiteOffWorld::countFallback(const std::string& r)
{
    if (counting()) fallbacks_[r]++;
}

void LiteOffWorld::countLlmCall(double lat, double wall)
{
    if (!counting()) return;
    ++llmCalls_;
    llmLatencySum_ += lat;
    llmWallSum_ += wall;
}

void LiteOffWorld::countDecision(double lat)
{
    if (!counting()) return;
    ++decisions_;
    decisionLatencySum_ += lat;
}

void LiteOffWorld::countEnergy(double c, double r, double i)
{
    if (!counting()) return;
    computeJ_ += c;
    radioJ_ += r;
    inferJ_ += i;
}

void LiteOffWorld::countReoffload()
{
    if (counting()) ++reoffloads_;
}

void LiteOffWorld::countModelUpdate()
{
    if (counting()) ++modelUpdates_;
}

void LiteOffWorld::taskResolved(const TaskRecord& r, TaskStatus st, double e2e)
{
    const Task& t = r.task;
    if (t.created < warmup_) return;
    ++generated_;
    int c = (int) t.cls;
    genByClass_[c]++;
    if (st == TaskStatus::InTime) {
        ++inTime_;
        inTimeByClass_[c]++;
    }
    else if (st == TaskStatus::Late) ++late_;
    else ++dropped_;
    if (st != TaskStatus::Dropped) {
        delaySum_ += e2e;
        ++delayN_;
        delayVec_.record(e2e);
        decSum_ += r.decisionDelay;
        txSum_ += r.txDelay;
        queueSum_ += r.queueDelay;
        execSum_ += r.execDelay;
        resSum_ += r.resultDelay;
        ++compN_;
    }
    if (r.offloaded) {
        ++offloaded_;
        if (st != TaskStatus::Dropped) ++offloadOk_;
    }
    if (r.oracleAgree) ++oracleAgree_;
    const char* sn = st == TaskStatus::InTime ? "in_time" : (st == TaskStatus::Late ? "late" : "dropped");
    if (trace_) {
        std::fprintf(trace_, "%ld,%d,%s,%.6f,%.4f,%.0f,%.0f,%s,%s,%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%s\n", t.id, t.origin, className(t.cls), t.created, t.tmax, t.bytes,
                     t.cycles, r.path.c_str(), r.targetKind.c_str(), r.split, sn, st == TaskStatus::Dropped ? -1.0 : e2e, r.decisionDelay, r.txDelay, r.queueDelay,
                     r.execDelay, r.resultDelay, r.oracleAgree ? 1 : 0, r.reoffloaded ? 1 : 0, r.fallbackReason.c_str());
    }
    if (dataset_ && !r.ctxJson.empty()) {
        std::fprintf(dataset_, "{\"scenario\":\"%s\",\"ctx\":%s,\"oracle\":{\"slot\":%d,\"split\":%d},\"chosen\":{\"slot\":%d,\"split\":%d,\"path\":\"%s\"},\"outcome\":{\"status\":\"%s\",\"delay_ms\":%.3f}}\n",
                     jsonEscape(scenarioTag_).c_str(), r.ctxJson.c_str(), r.oracleSlot, r.oracleSplit, r.chosenSlot, r.split, r.path.c_str(), sn,
                     st == TaskStatus::Dropped ? -1.0 : e2e * 1000);
    }
}

void LiteOffWorld::finish()
{
    // resolve tasks still pending in live vehicles (counted as dropped)
    std::vector<LiteOffNodeApp*> live;
    for (auto& kv : nodes_) live.push_back(kv.second);
    for (auto* n : live) n->flushPending();
    std::vector<double> served;
    for (auto* n : live)
        if (n->isRsu() || n->servedCycles() > 0) served.push_back(n->servedCycles());

    auto rec = [this](const char* n, double v) { recordScalar(n, v); };
    double g = std::max<long>(1, generated_);
    rec("tasksGenerated", generated_);
    rec("tasksInTime", inTime_);
    rec("tasksLate", late_);
    rec("tasksDropped", dropped_);
    rec("tcr", inTime_ / g);
    rec("lateRatio", late_ / g);
    rec("dropRatio", dropped_ / g);
    static const char* cn[] = {"Safety", "Perception", "Infotainment"};
    for (int c = 0; c < 3; ++c) {
        rec((std::string("tasksGenerated") + cn[c]).c_str(), genByClass_[c]);
        rec((std::string("tcr") + cn[c]).c_str(), genByClass_[c] ? (double) inTimeByClass_[c] / genByClass_[c] : NAN);
    }
    double n = std::max<long>(1, compN_);
    rec("meanDelay", delayN_ ? delaySum_ / delayN_ : NAN);
    rec("meanDecisionDelay", decSum_ / n);
    rec("meanTxDelay", txSum_ / n);
    rec("meanQueueDelay", queueSum_ / n);
    rec("meanExecDelay", execSum_ / n);
    rec("meanResultDelay", resSum_ / n);
    rec("tasksOffloaded", offloaded_);
    rec("offloadSuccessRate", offloaded_ ? (double) offloadOk_ / offloaded_ : NAN);
    rec("reoffloadRate", reoffloads_ / g);
    rec("oracleAgreement", oracleAgree_ / g);
    rec("decisions", decisions_);
    rec("meanDecisionLatency", decisions_ ? decisionLatencySum_ / decisions_ : 0);
    rec("llmCalls", llmCalls_);
    rec("meanLlmLatency", llmCalls_ ? llmLatencySum_ / llmCalls_ : 0);
    rec("meanLlmWallMs", llmCalls_ ? llmWallSum_ / llmCalls_ : 0);
    rec("modelUpdatesCompleted", modelUpdates_);
    rec("energyComputeJ", computeJ_);
    rec("energyRadioJ", radioJ_);
    rec("energyInferenceJ", inferJ_);
    rec("energyPerTaskJ", (computeJ_ + radioJ_ + inferJ_) / g);
    double total = 0, ctrl = 0;
    long uni = 0, fail = 0;
    for (int t = 0; t < (int) Traffic::N; ++t) {
        total += bytes_[t];
        if (t != (int) Traffic::Task && t != (int) Traffic::Result) ctrl += bytes_[t];
        uni += unicast_[t];
        fail += failed_[t];
        std::string nm = trafficName((Traffic) t);
        rec(("bytes_" + nm).c_str(), bytes_[t]);
        rec(("frames_" + nm).c_str(), frames_[t]);
        rec(("unicast_" + nm).c_str(), unicast_[t]);
        rec(("failed_" + nm).c_str(), failed_[t]);
    }
    rec("bytesTotal", total);
    rec("overheadRatio", total > 0 ? ctrl / total : 0);
    rec("packetDropRatio", uni ? (double) fail / uni : 0);
    rec("jainServerLoad", jainIndex(served));
    long pathTotal = 0;
    for (auto& kv : paths_) pathTotal += kv.second;
    for (const char* p : {"local", "rule", "cache", "llm", "rsu-llm", "fallback", "rl", "fixed", "oracle", "explore"}) {
        long v = paths_.count(p) ? paths_[p] : 0;
        rec((std::string("path_") + p).c_str(), v);
        rec((std::string("pathShare_") + p).c_str(), pathTotal ? (double) v / pathTotal : 0);
    }
    long fb = 0;
    for (auto& kv : fallbacks_) {
        rec(("fallback_" + kv.first).c_str(), kv.second);
        fb += kv.second;
    }
    rec("fallbackTotal", fb);
    rec("malformedRate", llmCalls_ ? (double) (fallbacks_.count("malformed") ? fallbacks_["malformed"] : 0) / llmCalls_ : 0);
    if (trace_) std::fclose(trace_);
    if (dataset_) std::fclose(dataset_);
    trace_ = dataset_ = nullptr;
    if (sidecarEnabled_ && sidecar_.connected()) {
        sidecar_.request("{\"op\":\"bye\"}");
        sidecar_.close();
    }
    instance_ = nullptr;
}

}  // namespace liteoff
