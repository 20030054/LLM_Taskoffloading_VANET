#include "LiteOffNodeApp.h"

#include "veins/base/phyLayer/PhyToMacControlInfo.h"
#include "veins/modules/messages/PhyControlMessage_m.h"
#include "veins/modules/phy/DeciderResult80211.h"

using namespace omnetpp;

namespace liteoff {

simsignal_t LiteOffNodeApp::sigChannelBusy_ = registerSignal("org_car2x_veins_modules_mac_sigChannelBusy");
simsignal_t LiteOffNodeApp::sigRetriesExceeded_ = registerSignal("org_car2x_veins_modules_mac_sigRetriesExceeded");

LiteOffNodeApp::~LiteOffNodeApp()
{
    cancelAndDelete(cpuDoneEvt_);
    cancelAndDelete(beaconEvt_);
}

void LiteOffNodeApp::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        world = LiteOffWorld::get();
        cpuHz_ = par("cpuHz").doubleValue();
        beaconBytes_ = par("beaconBytes").doubleValue();
        headerBytes_ = par("headerBytes").doubleValue();
        chunkBytes_ = par("chunkBytes").doubleValue();
        maxBeaconAge_ = par("maxBeaconAge").doubleValue();
        adaptiveMcs_ = par("adaptiveMcs").boolValue();
        dropExpired_ = par("dropExpired").boolValue();
        txPowerW_ = par("radioPowerW").doubleValue();
        kappa_ = par("kappa").doubleValue();
        est_.commRange = par("commRange").doubleValue();
        est_.macEfficiency = par("macEfficiency").doubleValue();
        est_.linkMargin = par("linkMargin").doubleValue();
        cpuDoneEvt_ = new cMessage("cpuDone", CPU_DONE_EVT);
        beaconEvt_ = new cMessage("loBeacon", BEACON_EVT);
        findHost()->subscribe(sigChannelBusy_, this);
        findHost()->subscribe(sigRetriesExceeded_, this);
        cbrWindowStart_ = loadWindowStart_ = simTime();
    }
    else if (stage == 1) {
        world->registerNode(myId, this);
        double bi = par("beaconInterval").doubleValue();
        scheduleAt(simTime() + uniform(0, bi), beaconEvt_);
    }
}

void LiteOffNodeApp::finish()
{
    flushPending();
    if (world) world->unregisterNode(myId);
    recordScalar("servedCycles", servedCycles_);
    recordScalar("energyUsedJ", energyUsedJ_);
}

Vec2 LiteOffNodeApp::truePos() const
{
    return {curPosition.x, curPosition.y};
}

Vec2 LiteOffNodeApp::trueVel() const
{
    return {curSpeed.x, curSpeed.y};
}

double LiteOffNodeApp::backlogSec() const
{
    double b = cpuBusy_ ? std::max(0.0, (currentEnd_ - simTime()).dbl()) : 0.0;
    for (const auto& j : cpuQueue_) b += j.cycles / cpuHz_;
    return b;
}

Neighbor* LiteOffNodeApp::neighbor(int addr)
{
    auto it = nbrs_.find(addr);
    return it == nbrs_.end() ? nullptr : &it->second;
}

double LiteOffNodeApp::estDistanceTo(int addr) const
{
    Vec2 me = truePos();
    auto it = nbrs_.find(addr);
    if (it != nbrs_.end()) {
        double age = (simTime() - it->second.lastRx).dbl();
        double x = it->second.pos.x + it->second.vel.x * age, y = it->second.pos.y + it->second.vel.y * age;
        return std::hypot(x - me.x, y - me.y);
    }
    if (auto* n = world->node(addr)) {
        Vec2 p = n->truePos();
        return std::hypot(p.x - me.x, p.y - me.y);
    }
    return est_.commRange;
}

int LiteOffNodeApp::mcsTo(int addr, double distHint) const
{
    auto it = nbrs_.find(addr);
    if (it != nbrs_.end() && (simTime() - it->second.lastRx).dbl() <= maxBeaconAge_) return std::min(mcsForSnr(it->second.rssi - noiseFloorDbm_, adaptiveMcs_), mcsForDistance(distHint, adaptiveMcs_));
    return mcsForDistance(distHint, adaptiveMcs_);
}

// ------------------------------------------------------------------ transmission
void LiteOffNodeApp::sendFrame(veins::BaseFrame1609_4* f, int dest, Traffic cat, double bytes, double distHint, simtime_t delay)
{
    populateWSM(f, dest >= 0 ? dest : veins::LAddress::L2BROADCAST());
    double total = bytes + headerBytes_;
    f->setByteLength((int64_t) std::ceil(total));
    int mcs = (cat == Traffic::Beacon || dest < 0) ? 2 : mcsTo(dest, distHint);
    if (cat == Traffic::Beacon) f->setUserPriority(7);
    else if (cat == Traffic::Task || cat == Traffic::Model || cat == Traffic::Fed) f->setUserPriority(4);
    else f->setUserPriority(6);
    auto* ctrl = new veins::PhyControlMessage();
    ctrl->setMcs(mcs);
    f->setControlInfo(ctrl);
    world->countTx(cat, total, dest >= 0);
    chargeRadio(total, rateForMcs(mcs));
    if (delay > 0) sendDelayedDown(f, delay);
    else sendDown(f);
}

void LiteOffNodeApp::chargeRadio(double bytes, double rateBps)
{
    if (!chargeEnergy_) return;
    double j = txPowerW_ * bytes * 8.0 / rateBps;
    energyUsedJ_ += j;
    world->countEnergy(0, j, 0);
}

void LiteOffNodeApp::chargeCompute(double cycles)
{
    if (!chargeEnergy_) return;
    double j = kappa_ * cycles * cpuHz_ * cpuHz_;
    energyUsedJ_ += j;
    world->countEnergy(j, 0, 0);
}

void LiteOffNodeApp::fillBeacon(LoBeacon* b)
{
    Vec2 p = truePos(), v = trueVel();
    b->setSrc(myId);
    b->setIsRsu(isRsu());
    b->setSeq(beaconSeq_++);
    b->setX(p.x);
    b->setY(p.y);
    b->setVx(v.x);
    b->setVy(v.y);
    b->setCpuHz(cpuHz_);
    b->setBacklogSec(backlogSec());
    b->setQueueLen(queueLen());
    b->setLoad(load_);
    b->setEnergy(1.0);
    b->setCbr(cbr_);
}

// ------------------------------------------------------------------ events
void LiteOffNodeApp::handleSelfMsg(cMessage* msg)
{
    switch (msg->getKind()) {
    case BEACON_EVT: {
        // update windowed channel-busy ratio and CPU load
        simtime_t now = simTime();
        double win = (now - cbrWindowStart_).dbl();
        if (win > 0) {
            double busy = chBusyAccum_ + (chBusy_ ? (now - std::max(chBusySince_, cbrWindowStart_)).dbl() : 0.0);
            cbr_ = 0.5 * cbr_ + 0.5 * std::clamp(busy / win, 0.0, 1.0);
            double cbusy = cpuBusyAccum_ + (cpuBusy_ ? (now - std::max(cpuBusySince_, loadWindowStart_)).dbl() : 0.0);
            load_ = std::clamp(cbusy / win, 0.0, 1.0);
        }
        chBusyAccum_ = 0;
        cpuBusyAccum_ = 0;
        cbrWindowStart_ = loadWindowStart_ = now;
        auto* b = new LoBeacon("beacon");
        fillBeacon(b);
        sendFrame(b, -1, Traffic::Beacon, beaconBytes_, 0);
        scheduleAt(now + par("beaconInterval").doubleValue(), beaconEvt_);
        // garbage-collect stale partial receptions
        for (auto it = rx_.begin(); it != rx_.end();) {
            if (it->second.deadline + 2 < now) it = rx_.erase(it);
            else ++it;
        }
        break;
    }
    case CPU_DONE_EVT: {
        cpuBusyAccum_ += (simTime() - std::max(cpuBusySince_, loadWindowStart_)).dbl();
        cpuBusy_ = false;
        Job j = current_;
        if (j.type == Job::Served) servedCycles_ += j.cycles;
        startNextJob();
        onJobDone(j);
        break;
    }
    default:
        DemoBaseApplLayer::handleSelfMsg(msg);
    }
}

void LiteOffNodeApp::enqueueJob(const Job& j, bool front)
{
    Job c = j;
    c.enqueued = simTime();
    if (front) cpuQueue_.push_front(c);
    else cpuQueue_.push_back(c);
    if (!cpuBusy_) startNextJob();
}

void LiteOffNodeApp::startNextJob()
{
    while (!cpuQueue_.empty() && !cpuBusy_) {
        Job j = cpuQueue_.front();
        cpuQueue_.pop_front();
        if (dropExpired_ && j.type != Job::Inference && j.deadline < simTime()) {
            EV_DETAIL << "dropping expired job " << j.taskId << endl;
            continue;  // origin will time out
        }
        j.started = simTime();
        current_ = j;
        cpuBusy_ = true;
        cpuBusySince_ = simTime();
        currentEnd_ = simTime() + j.cycles / cpuHz_;
        if (j.type != Job::Inference) chargeCompute(j.cycles);
        scheduleAt(currentEnd_, cpuDoneEvt_);
    }
}

void LiteOffNodeApp::onJobDone(const Job& j)
{
    if (j.type == Job::Served) transmitResult(j.origin, j.taskId, j.attempt, isRsu() ? (int) NodeKind::Rsu : (int) NodeKind::Vehicle, j.resultBytes, {j.enqueued, j.started, simTime()});
}

void LiteOffNodeApp::onTaskReceived(int origin, long taskId, int attempt, const RxPartial& p)
{
    Job j;
    j.type = Job::Served;
    j.taskId = taskId;
    j.attempt = attempt;
    j.origin = origin;
    j.cycles = p.cycles;
    j.resultBytes = p.resultBytes;
    j.deadline = p.deadline;
    enqueueJob(j);
}

static LoResult* makeResult(int src, int origin, long taskId, int attempt, int executorKind, const LiteOffNodeApp::Timing& tm)
{
    auto* r = new LoResult("result");
    r->setSrc(src);
    r->setTaskId(taskId);
    r->setAttempt(attempt);
    r->setOrigin(origin);
    r->setExecutor(src);
    r->setExecutorKind(executorKind);
    r->setRxDone(tm.rxDone);
    r->setExecStart(tm.execStart);
    r->setExecEnd(tm.execEnd);
    return r;
}

void LiteOffNodeApp::transmitResult(int origin, long taskId, int attempt, int executorKind, double resultBytes, const Timing& tm)
{
    Enter_Method_Silent();
    auto* r = makeResult(myId, origin, taskId, attempt, executorKind, tm);
    if (!isRsu()) {
        sendFrame(r, origin, Traffic::Result, resultBytes, estDistanceTo(origin));
        return;
    }
    // RSU: deliver from the RSU closest to the origin's current position (wired backhaul relay)
    LiteOffNodeApp* dst = world->node(origin);
    if (!dst) {
        delete r;
        return;
    }
    Vec2 p = dst->truePos();
    double d = 0;
    LiteOffNodeApp* best = world->nearestRsu(p.x, p.y, &d);
    if (!best || d > est_.commRange) {
        delete r;  // origin outside all coverage: result lost
        return;
    }
    if (best == this) {
        sendFrame(r, origin, Traffic::Result, resultBytes, d);
    }
    else {
        delete r;
        double hop = par("backhaulHopDelay").doubleValue();
        best->relayResult(origin, taskId, attempt, executorKind, resultBytes, tm, hop);
    }
}

void LiteOffNodeApp::relayResult(int origin, long taskId, int attempt, int executorKind, double resultBytes, const Timing& tm, double delay)
{
    Enter_Method_Silent();
    auto* r = makeResult(myId, origin, taskId, attempt, executorKind, tm);
    sendFrame(r, origin, Traffic::Result, resultBytes, estDistanceTo(origin), delay);
}

// ------------------------------------------------------------------ reception
void LiteOffNodeApp::onWSM(veins::BaseFrame1609_4* wsm)
{
    if (auto* b = dynamic_cast<LoBeacon*>(wsm)) {
        Neighbor& n = nbrs_[b->getSrc()];
        if (n.lastSeq >= 0) {
            int gap = std::max(0, b->getSeq() - n.lastSeq - 1);
            n.loss = 0.8 * n.loss + 0.2 * (double) gap / (gap + 1);
        }
        n.addr = b->getSrc();
        n.isRsu = b->isRsu();
        n.pos = {b->getX(), b->getY()};
        n.vel = {b->getVx(), b->getVy()};
        n.cpuHz = b->getCpuHz();
        n.backlogSec = b->getBacklogSec();
        n.queueLen = b->getQueueLen();
        n.load = b->getLoad();
        n.energy = b->getEnergy();
        n.cbr = b->getCbr();
        n.modelVersion = b->getModelVersion();
        n.modelRecvVersion = b->getModelRecvVersion();
        n.modelRecvChunks = b->getModelRecvChunks();
        n.nextEdges = b->getNextEdges();
        n.lastRx = simTime();
        n.lastSeq = b->getSeq();
        if (auto* res = dynamic_cast<veins::DeciderResult80211*>(veins::PhyToMacControlInfo::getDeciderResult(wsm))) n.rssi = res->getRecvPower_dBm();
        onBeacon(b);
    }
    else if (auto* c = dynamic_cast<LoTaskChunk*>(wsm)) {
        auto key = std::make_tuple(c->getOrigin(), c->getTaskId(), c->getAttempt());
        RxPartial& p = rx_[key];
        if (p.nChunks == 0) {
            p.nChunks = c->getNChunks();
            p.partBytes = c->getPartBytes();
            p.cycles = c->getCycles();
            p.resultBytes = c->getResultBytes();
            p.created = c->getCreated();
            p.deadline = c->getDeadline();
            p.toCloud = c->getToCloud();
            p.firstRx = simTime();
        }
        p.got.insert(c->getSeq());
        if ((int) p.got.size() >= p.nChunks) {
            RxPartial done = p;
            rx_.erase(key);
            onTaskReceived(c->getOrigin(), c->getTaskId(), c->getAttempt(), done);
        }
    }
    else if (auto* r = dynamic_cast<LoResult*>(wsm)) onResult(r);
    else if (auto* m = dynamic_cast<LoModelChunk*>(wsm)) onModelChunk(m);
    else if (auto* q = dynamic_cast<LoLlmQuery*>(wsm)) onLlmQuery(q);
    else if (auto* a = dynamic_cast<LoLlmReply*>(wsm)) onLlmReply(a);
}

void LiteOffNodeApp::receiveSignal(cComponent* source, simsignal_t signalID, bool b, cObject* details)
{
    Enter_Method_Silent();
    if (signalID != sigChannelBusy_) return;
    simtime_t now = simTime();
    if (b && !chBusy_) {
        chBusy_ = true;
        chBusySince_ = now;
    }
    else if (!b && chBusy_) {
        chBusy_ = false;
        chBusyAccum_ += (now - std::max(chBusySince_, cbrWindowStart_)).dbl();
    }
}

void LiteOffNodeApp::receiveSignal(cComponent* source, simsignal_t signalID, cObject* obj, cObject* details)
{
    if (signalID == sigRetriesExceeded_) {
        Enter_Method_Silent();
        auto* f = dynamic_cast<veins::BaseFrame1609_4*>(obj);
        if (!f) return;
        Traffic cat = Traffic::LlmCtl;
        if (dynamic_cast<LoTaskChunk*>(f)) cat = Traffic::Task;
        else if (dynamic_cast<LoResult*>(f)) cat = Traffic::Result;
        else if (auto* m = dynamic_cast<LoModelChunk*>(f)) cat = m->getFederated() ? Traffic::Fed : Traffic::Model;
        world->countTxFailed(cat);
        onUnicastFailed(f);
        return;
    }
    DemoBaseApplLayer::receiveSignal(source, signalID, obj, details);
}

}  // namespace liteoff
