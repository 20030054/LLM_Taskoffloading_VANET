#pragma once

#include "LiteOffNodeApp.h"

namespace liteoff {

// RSU with MEC server: executes offloaded tasks, forwards to cloud, hosts the RSU-side LLM,
// manages model versions (adapter push) and B4 federated aggregation.
class RsuApp : public LiteOffNodeApp {
public:
    ~RsuApp() override;
    bool isRsu() const override { return true; }

protected:
    enum RKinds { CLOUD_DONE_EVT = 300, MODEL_PUSH_EVT, FED_AGG_EVT };

    struct CloudJob {
        int origin;
        long taskId;
        int attempt;
        double resultBytes;
        Timing tm;
    };

    void initialize(int stage) override;
    void finish() override;
    void handleSelfMsg(omnetpp::cMessage* msg) override;
    void onTaskReceived(int origin, long taskId, int attempt, const RxPartial& p) override;
    void onLlmQuery(LoLlmQuery* q) override;
    void onUnicastFailed(veins::BaseFrame1609_4* frame) override;
    int currentVersion() const;
    void pushModel();

    double cloudHz_ = 100e9, cloudDelay_ = 0.05, backhaulRate_ = 1e9;
    double rsuSpeedup_ = 4;
    std::string rsuLatKey_;
    omnetpp::simtime_t llmBusyUntil_;
    std::map<long, CloudJob> cloud_;
    long cloudRef_ = 0;
    // model manager
    bool modelMgmt_ = false;
    double adapterBytes_ = 4e6, updateInterval_ = 0, pushPeriod_ = 0.05, pushCbr_ = 0.4;
    int chunksPerPush_ = 2;
    std::map<int, std::pair<int, int>> sent_;  // vehicle -> (version, next seq)
    int lastServed_ = -1;
    omnetpp::cMessage* pushEvt_ = nullptr;
    omnetpp::cMessage* fedEvt_ = nullptr;
    double fedPeriod_ = 10, fedModelBytes_ = 50000;
    long llmQueries_ = 0;
};

}  // namespace liteoff
