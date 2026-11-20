#pragma once

#include <omnetpp.h>

#include <cstdio>
#include <map>
#include <string>

#include "core/Core.h"

namespace liteoff {

class LiteOffNodeApp;

// Byte/frame accounting categories
enum class Traffic { Beacon = 0, Task, Result, LlmCtl, Model, Fed, N };
const char* trafficName(Traffic t);

// Final status of a task
enum class TaskStatus { InTime, Late, Dropped };

struct TaskRecord {
    Task task;
    std::string path;        // decision path of the (first) decision
    std::string targetKind;  // local|veh|rsu|cloud
    int split = 0;
    bool oracleAgree = false;
    bool reoffloaded = false;
    bool offloaded = false;
    double decisionDelay = 0;
    // delay components of the remote part (if any)
    double txDelay = 0, queueDelay = 0, execDelay = 0, resultDelay = 0;
    std::string ctxJson;  // for dataset logging (empty if disabled)
    int oracleSlot = 0, oracleSplit = 0, chosenSlot = 0;
    std::string fallbackReason;
};

// Blocking JSON-lines TCP client to the Python side-car.
class SidecarClient {
public:
    ~SidecarClient();
    bool connect(const std::string& host, int port, double timeoutSec, std::string* err);
    bool connected() const { return fd_ >= 0; }
    // sends one line, returns reply line (empty on error)
    std::string request(const std::string& line, double* wallMs = nullptr);
    void close();

private:
    int fd_ = -1;
    std::string buf_;
};

// Global singleton module: registry, metrics, side-car, latency model, logs.
class LiteOffWorld : public omnetpp::cSimpleModule {
public:
    static LiteOffWorld* get();

    // ---------- registry (for oracle / RSU relay)
    void registerNode(int addr, LiteOffNodeApp* app);
    void unregisterNode(int addr);
    LiteOffNodeApp* node(int addr) const;
    const std::map<int, LiteOffNodeApp*>& nodes() const { return nodes_; }
    LiteOffNodeApp* nearestRsu(double x, double y, double* dist = nullptr) const;

    // ---------- configuration accessors
    double warmup() const { return warmup_; }
    bool counting() const { return omnetpp::simTime().dbl() >= warmup_; }
    const LatencyModel& latency() const { return latency_; }
    SidecarClient& sidecar();
    bool sidecarAvailable() const { return sidecarEnabled_; }
    std::string scenarioTag() const { return scenarioTag_; }

    // ---------- metrics
    void countTx(Traffic t, double bytes, bool unicast);
    void countTxFailed(Traffic t);
    void countPath(const std::string& path);
    void countFallback(const std::string& reason);
    void countLlmCall(double modelLatency, double wallMs);
    void countDecision(double latency);
    void countEnergy(double computeJ, double radioJ, double inferJ);
    void countReoffload();
    void countModelUpdate();
    void taskResolved(const TaskRecord& r, TaskStatus st, double endToEnd);

    // ---------- RSU-hosted LLM query store (simulation shortcut, see LoLlmQuery)
    long storeQuery(const std::string& requestJson);
    std::string takeQuery(long id);

protected:
    int numInitStages() const override { return 1; }
    void initialize() override;
    void handleMessage(omnetpp::cMessage* msg) override;
    void finish() override;

private:
    static LiteOffWorld* instance_;
    std::map<int, LiteOffNodeApp*> nodes_;
    double warmup_ = 0;
    LatencyModel latency_;
    SidecarClient sidecar_;
    bool sidecarEnabled_ = false;
    bool sidecarTried_ = false;
    std::string sidecarHost_;
    int sidecarPort_ = 0;
    std::string scenarioTag_;
    FILE* trace_ = nullptr;
    FILE* dataset_ = nullptr;
    std::map<long, std::string> queries_;
    long nextQuery_ = 1;

    // counters
    long generated_ = 0, inTime_ = 0, late_ = 0, dropped_ = 0;
    long genByClass_[3] = {0, 0, 0}, inTimeByClass_[3] = {0, 0, 0};
    double delaySum_ = 0;
    long delayN_ = 0;
    double decSum_ = 0, txSum_ = 0, queueSum_ = 0, execSum_ = 0, resSum_ = 0;
    long compN_ = 0;
    long offloaded_ = 0, offloadOk_ = 0, reoffloads_ = 0, oracleAgree_ = 0, decisions_ = 0;
    double decisionLatencySum_ = 0;
    long llmCalls_ = 0;
    double llmLatencySum_ = 0, llmWallSum_ = 0;
    long modelUpdates_ = 0;
    double computeJ_ = 0, radioJ_ = 0, inferJ_ = 0;
    double bytes_[(int) Traffic::N] = {0};
    long frames_[(int) Traffic::N] = {0};
    long unicast_[(int) Traffic::N] = {0};
    long failed_[(int) Traffic::N] = {0};
    std::map<std::string, long> paths_, fallbacks_;
    omnetpp::cOutVector delayVec_;
};

}  // namespace liteoff
