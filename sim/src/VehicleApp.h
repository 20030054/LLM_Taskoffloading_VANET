#pragma once

#include <random>
#include <set>

#include "LiteOffNodeApp.h"

namespace liteoff {

// Vehicle: task generator + offloading decision pipeline + execution server for neighbours.
class VehicleApp : public LiteOffNodeApp {
public:
    ~VehicleApp() override;
    bool isRsu() const override { return false; }
    Vec2 truePos() const override;
    Vec2 trueVel() const override;
    void flushPending() override;

protected:
    enum VKinds { GEN_TASK_EVT = 200, DECISION_EVT, WATCHDOG_EVT, FINAL_EVT, QUERY_TIMEOUT_EVT, FED_EVT, UPLINK_EVT };

    struct Pending {
        Task task;
        TaskRecord rec;
        int attempt = 0;
        bool decided = false;
        bool localNeeded = false, localDone = false;
        bool remoteNeeded = false, remoteDone = false;
        double remoteFrac = 0;
        int target = -1;
        NodeKind targetKind = NodeKind::Local;
        bool reoffloaded = false;
        std::set<int> failedTargets;
        omnetpp::simtime_t sendTime;
        omnetpp::simtime_t decisionStart;
        omnetpp::cMessage* watchdog = nullptr;
        omnetpp::cMessage* finalEvt = nullptr;
        omnetpp::cMessage* decisionEvt = nullptr;
        omnetpp::cMessage* queryTimeout = nullptr;
        omnetpp::cMessage* uplinkEvt = nullptr;
        // paced upload of the current attempt
        int upNext = 0, upN = 0, upMcs = 2;
        double upBytes = 0, upFrac = 1;
        long queryId = 0;
        Decision pendingDecision;
        Context actCtx;  // context whose slots the decision refers to
        bool rlDecision = false;
    };

    void initialize(int stage) override;
    void handleSelfMsg(omnetpp::cMessage* msg) override;
    void fillBeacon(LoBeacon* b) override;
    void onResult(LoResult* r) override;
    void onModelChunk(LoModelChunk* m) override;
    void onLlmReply(LoLlmReply* r) override;
    void onJobDone(const Job& j) override;
    void onTaskReceived(int origin, long taskId, int attempt, const RxPartial& p) override;
    void onUnicastFailed(veins::BaseFrame1609_4* frame) override;

    // pipeline
    Task generateTask();
    void startDecision(Pending& p, bool reoffload);
    Context buildContext(const Task& t, const std::set<int>& exclude, bool rich, int K = -1);
    Context trueContext(const Context& ctx);
    Candidate makeCandidate(int addr, bool isRsu, Vec2 pos, Vec2 vel, double cpuHz, double backlog, double load, int queueLen, double energy);
    void decideLiteOff(Pending& p, const Context& ctx);
    void decideRl(Pending& p, const Context& ctx);
    void decideViaRsu(Pending& p, const Context& ctx, const std::string& model, const std::string& families, const std::string& scheme);
    void applyAfter(Pending& p, const Context& ctx, Decision d, double delay);
    void fallback(Pending& p, const Context& ctx, const std::string& reason, double delay);
    void act(Pending& p);
    void sendTaskPart(Pending& p, const Candidate& c, double frac);
    void pumpUplink(Pending& p);
    void remoteFailed(Pending& p, const std::string& why);
    void checkComplete(Pending& p);
    void resolve(Pending& p, TaskStatus st);
    void cancelEvents(Pending& p);
    void updateTrust(int addr, bool ok);
    std::string sidecarDecide(const std::string& scheme, const std::string& model, const Context& ctx, const std::string& families, std::map<std::string, std::string>& reply, double* wallMs);
    void chargeInference(double seconds, const std::string& model);
    int nearestRsuRegion();
    std::string nextEdges();

    // params
    std::string scheme_, mode_, families_, llmModel_, latKey_, rsuModel_;
    int K_ = 5;
    GateParams gp_;
    RuleWeights rw_;
    bool validatorOn_ = true, ignoreLlmLatency_ = false, llmOnCpu_ = false, reoffloadOn_ = true, cloudOn_ = true;
    bool escalateCritical_ = false, richContext_ = true, routeInBeacons_ = true, modelMgmt_ = true;
    double confThreshold_ = 0.5, llmSpeedup_ = 1, llmMaxFrac_ = 0.5, lateGrace_ = 1, watchdogFactor_ = 2, minWatchdog_ = 0.05;
    double cloudHz_ = 100e9, cloudDelay_ = 0.05, taskRate_ = 1, taskStop_ = 1e9, energyBudget_ = 5000;
    double exploreOracle_ = 0.5, exploreRandom_ = 0.25, queryBytesFactor_ = 0.5;
    double fedPeriod_ = 10, fedModelBytes_ = 50000;
    int uplinkBurst_ = 2;
    double uplinkPacing_ = 1.0;
    std::string rlMode_;

    // private workload RNG (identical task streams across schemes for paired comparison)
    std::mt19937_64 wrng_;
    double wuni(double a, double b) { return std::uniform_real_distribution<double>(a, b)(wrng_); }
    double wexp(double mean) { return std::exponential_distribution<double>(1.0 / mean)(wrng_); }

    // state
    std::map<long, Pending> pend_;
    std::map<long, long> queryToTask_;
    std::map<int, double> trust_;
    std::map<int, double> timeouts_;
    DecisionCache cache_;
    long taskCounter_ = 0;
    omnetpp::simtime_t accelBusyUntil_;
    omnetpp::cMessage* genEvt_ = nullptr;
    omnetpp::cMessage* fedEvt_ = nullptr;
    int modelVersion_ = 0, modelRecvVersion_ = 0, modelRecvChunks_ = 0;
    std::string roadId_;
    bool selfish_ = false;
};

}  // namespace liteoff
