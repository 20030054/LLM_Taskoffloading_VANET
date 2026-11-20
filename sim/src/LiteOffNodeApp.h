#pragma once

#include <deque>
#include <map>
#include <set>
#include <string>
#include <tuple>

#include "veins/modules/application/ieee80211p/DemoBaseApplLayer.h"

#include "LiteOffWorld.h"
#include "core/Core.h"
#include "messages/LiteOffMsgs_m.h"

namespace liteoff {

// Beacon-derived knowledge about a neighbour (possibly stale).
struct Neighbor {
    int addr = -1;
    bool isRsu = false;
    Vec2 pos, vel;
    double cpuHz = 1e9, backlogSec = 0, load = 0, energy = 1, cbr = 0;
    int queueLen = 0;
    int modelVersion = 0, modelRecvVersion = 0, modelRecvChunks = 0;
    std::string nextEdges;
    omnetpp::simtime_t lastRx;
    int lastSeq = -1;
    double loss = 0;
    double rssi = -90;
};

// Base class for vehicles and RSUs: everything a node needs to act as an execution server.
class LiteOffNodeApp : public veins::DemoBaseApplLayer {
public:
    ~LiteOffNodeApp() override;

    // ---- true state (used by the full-information oracle and RSU relay)
    virtual bool isRsu() const = 0;
    virtual Vec2 truePos() const;
    virtual Vec2 trueVel() const;
    double cpuHz() const { return cpuHz_; }
    double backlogSec() const;
    int queueLen() const { return (int) cpuQueue_.size() + (cpuBusy_ ? 1 : 0); }
    double channelBusyRatio() const { return cbr_; }
    double servedCycles() const { return servedCycles_; }
    int address() const { return myId; }
    double commRange() const { return est_.commRange; }

    // called by the world at the end of the run (resolve pending tasks)
    virtual void flushPending() {}
    // RSU relay of a result toward a vehicle (backhaul hop)
    struct Timing {
        omnetpp::simtime_t rxDone, execStart, execEnd;
    };
    void transmitResult(int origin, long taskId, int attempt, int executorKind, double resultBytes, const Timing& tm);
    void relayResult(int origin, long taskId, int attempt, int executorKind, double resultBytes, const Timing& tm, double delay);

protected:
    enum Kinds { BEACON_EVT = 100, CPU_DONE_EVT, NODE_LAST_EVT };

    struct Job {
        enum Type { LocalPart = 0, Served = 1, Inference = 2 } type = LocalPart;
        long taskId = 0;
        int attempt = 0;
        int origin = -1;
        double cycles = 0;
        double resultBytes = 0;
        omnetpp::simtime_t deadline;
        omnetpp::simtime_t enqueued;
        omnetpp::simtime_t started;
        bool toCloud = false;
        long ref = 0;  // inference: decision reference
    };

    struct RxPartial {
        int nChunks = 0;
        std::set<int> got;
        double partBytes = 0, cycles = 0, resultBytes = 0;
        omnetpp::simtime_t created, deadline, firstRx;
        bool toCloud = false;
    };

    void initialize(int stage) override;
    void finish() override;
    void handleSelfMsg(omnetpp::cMessage* msg) override;
    void onWSM(veins::BaseFrame1609_4* wsm) override;
    void receiveSignal(omnetpp::cComponent* source, omnetpp::simsignal_t signalID, bool b, omnetpp::cObject* details) override;
    void receiveSignal(omnetpp::cComponent* source, omnetpp::simsignal_t signalID, omnetpp::cObject* obj, omnetpp::cObject* details) override;
    using veins::DemoBaseApplLayer::receiveSignal;

    // ---- hooks for subclasses
    virtual void onBeacon(LoBeacon* b) {}
    virtual void onResult(LoResult* r) {}
    virtual void onModelChunk(LoModelChunk* m) {}
    virtual void onLlmQuery(LoLlmQuery* q) {}
    virtual void onLlmReply(LoLlmReply* r) {}
    virtual void onJobDone(const Job& j);
    virtual void onTaskReceived(int origin, long taskId, int attempt, const RxPartial& p);
    virtual void onUnicastFailed(veins::BaseFrame1609_4* frame) {}
    virtual void fillBeacon(LoBeacon* b);

    // ---- helpers
    void sendFrame(veins::BaseFrame1609_4* f, int dest, Traffic cat, double bytes, double distHint, omnetpp::simtime_t delay = 0);
    double airtime(double bytes, int mcs) const { return (bytes + headerBytes_ + 40) * 8.0 / rateForMcs(mcs) + 150e-6; }
    void enqueueJob(const Job& j, bool front = false);
    void startNextJob();
    double estDistanceTo(int addr) const;
    // MCS for a unicast to addr: SNR of its last beacon if known, else distance based
    int mcsTo(int addr, double distHint) const;
    double noiseFloorDbm_ = -98;
    Neighbor* neighbor(int addr);
    double beaconAge(const Neighbor& n) const { return (omnetpp::simTime() - n.lastRx).dbl(); }
    void chargeRadio(double bytes, double rateBps);
    void chargeCompute(double cycles);
    LiteOffWorld* world = nullptr;

    // parameters
    double cpuHz_ = 2e9;
    double beaconBytes_ = 200;
    double headerBytes_ = 40;
    double chunkBytes_ = 8000;
    double maxBeaconAge_ = 2.5;
    bool adaptiveMcs_ = false;
    bool dropExpired_ = true;
    double txPowerW_ = 1.0;
    double kappa_ = 1e-28;
    bool chargeEnergy_ = true;
    EstParams est_;

    // state
    std::map<int, Neighbor> nbrs_;
    std::deque<Job> cpuQueue_;
    bool cpuBusy_ = false;
    Job current_;
    omnetpp::simtime_t currentEnd_;
    omnetpp::cMessage* cpuDoneEvt_ = nullptr;
    omnetpp::cMessage* beaconEvt_ = nullptr;
    std::map<std::tuple<int, long, int>, RxPartial> rx_;
    int beaconSeq_ = 0;
    double servedCycles_ = 0;
    // channel busy ratio estimation
    bool chBusy_ = false;
    omnetpp::simtime_t chBusySince_, cbrWindowStart_;
    double chBusyAccum_ = 0;
    double cbr_ = 0;
    // cpu load estimation
    double cpuBusyAccum_ = 0;
    omnetpp::simtime_t cpuBusySince_, loadWindowStart_;
    double load_ = 0;
    // energy
    double energyUsedJ_ = 0;

    static omnetpp::simsignal_t sigChannelBusy_;
    static omnetpp::simsignal_t sigRetriesExceeded_;
};

}  // namespace liteoff
