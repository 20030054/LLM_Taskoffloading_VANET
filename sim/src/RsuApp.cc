#include "RsuApp.h"

#include <cstdint>
#include <sstream>

using namespace omnetpp;

namespace liteoff {

Define_Module(RsuApp);

RsuApp::~RsuApp()
{
    for (auto* m : {pushEvt_, fedEvt_}) cancelAndDelete(m);
}

void RsuApp::initialize(int stage)
{
    LiteOffNodeApp::initialize(stage);
    if (stage != 0) return;
    chargeEnergy_ = false;  // RSUs are mains powered
    cModule* w = world;
    cloudHz_ = w->par("cloudHz").doubleValue();
    cloudDelay_ = w->par("cloudDelay").doubleValue();
    backhaulRate_ = par("backhaulRate").doubleValue();
    rsuSpeedup_ = w->par("rsuLlmSpeedup").doubleValue();
    rsuLatKey_ = w->par("rsuLlmLatencyKey").stdstringValue();
    if (rsuLatKey_.empty()) rsuLatKey_ = w->par("rsuLlmModel").stdstringValue();
    std::string scheme = w->par("scheme").stdstringValue();
    modelMgmt_ = w->par("modelManagement").boolValue() && scheme == "liteoff" && w->par("deploymentMode").stdstringValue() != "rsu";
    adapterBytes_ = w->par("adapterBytes").doubleValue();
    updateInterval_ = w->par("modelUpdateInterval").doubleValue();
    pushPeriod_ = w->par("modelPushPeriod").doubleValue();
    pushCbr_ = w->par("pushCbrThreshold").doubleValue();
    chunksPerPush_ = w->par("chunksPerPush").intValue();
    fedPeriod_ = w->par("fedPeriod").doubleValue();
    fedModelBytes_ = w->par("fedModelBytes").doubleValue();
    if (modelMgmt_) {
        pushEvt_ = new cMessage("modelPush", MODEL_PUSH_EVT);
        scheduleAt(simTime() + uniform(0, pushPeriod_), pushEvt_);
    }
    if (scheme == "b4") {
        fedEvt_ = new cMessage("fedAgg", FED_AGG_EVT);
        scheduleAt(simTime() + fedPeriod_ + uniform(0, 1), fedEvt_);
    }
}

void RsuApp::finish()
{
    LiteOffNodeApp::finish();
    recordScalar("llmQueries", llmQueries_);
}

int RsuApp::currentVersion() const
{
    if (updateInterval_ <= 0) return 1;
    return 1 + (int) std::floor(simTime().dbl() / updateInterval_);
}

void RsuApp::handleSelfMsg(cMessage* msg)
{
    switch (msg->getKind()) {
    case CLOUD_DONE_EVT: {
        long ref = (long) (intptr_t) msg->getContextPointer();
        delete msg;
        auto it = cloud_.find(ref);
        if (it == cloud_.end()) return;
        CloudJob j = it->second;
        cloud_.erase(it);
        transmitResult(j.origin, j.taskId, j.attempt, (int) NodeKind::Cloud, j.resultBytes, j.tm);
        return;
    }
    case MODEL_PUSH_EVT:
        pushModel();
        scheduleAt(simTime() + pushPeriod_, pushEvt_);
        return;
    case FED_AGG_EVT: {
        std::ostringstream o;
        o << "{\"op\":\"fedavg\",\"region\":" << myId << ",\"t\":" << simTime().dbl() << "}";
        if (world->sidecarAvailable()) world->sidecar().request(o.str());
        int n = (int) std::ceil(fedModelBytes_ / chunkBytes_);
        for (int i = 0; i < n; ++i) {
            auto* m = new LoModelChunk("fedDown");
            m->setSrc(myId);
            m->setFederated(true);
            m->setSeq(i);
            m->setNChunks(n);
            sendFrame(m, -1, Traffic::Fed, std::min(chunkBytes_, fedModelBytes_ - i * chunkBytes_), 0);
        }
        scheduleAt(simTime() + fedPeriod_, fedEvt_);
        return;
    }
    default:
        LiteOffNodeApp::handleSelfMsg(msg);
    }
}

void RsuApp::onTaskReceived(int origin, long taskId, int attempt, const RxPartial& p)
{
    if (!p.toCloud) {
        LiteOffNodeApp::onTaskReceived(origin, taskId, attempt, p);
        return;
    }
    // forward over the wired backhaul, execute in the cloud, come back
    double up = cloudDelay_ + p.partBytes * 8.0 / backhaulRate_;
    double exec = p.cycles / cloudHz_;
    double down = cloudDelay_ + p.resultBytes * 8.0 / backhaulRate_;
    long ref = ++cloudRef_;
    CloudJob j{origin, taskId, attempt, p.resultBytes, {simTime(), simTime() + up, simTime() + up + exec}};
    cloud_[ref] = j;
    auto* m = new cMessage("cloudDone", CLOUD_DONE_EVT);
    m->setContextPointer((void*) (intptr_t) ref);
    scheduleAt(simTime() + up + exec + down, m);
}

void RsuApp::onLlmQuery(LoLlmQuery* q)
{
    std::string req = world->takeQuery(q->getQueryId());
    if (req.empty()) return;
    ++llmQueries_;
    double wall = 0;
    std::string line = world->sidecar().request(req, &wall);
    bool ok = false;
    auto rep = parseFlatJson(line, &ok);
    auto* r = new LoLlmReply("llmReply");
    r->setSrc(myId);
    r->setQueryId(q->getQueryId());
    double L = 0;
    if (ok && !rep.count("error")) {
        ParsedAnswer a = parseAnswer(rep.count("raw") ? rep["raw"] : "");
        r->setOk(a.ok);
        r->setSlot(a.slot);
        r->setSplit(a.split);
        r->setConf(rep.count("conf") && rep["conf"] != "null" ? std::stod(rep["conf"]) : -1);
        int nIn = std::stoi(rep["n_in"]), nOut = std::stoi(rep["n_out"]);
        r->setNIn(nIn);
        r->setNOut(nOut);
        L = world->latency().latency(rsuLatKey_, nIn, nOut, rsuSpeedup_);
        world->countLlmCall(L, wall);
    }
    else {
        r->setOk(false);
    }
    simtime_t start = std::max(simTime(), llmBusyUntil_);
    llmBusyUntil_ = start + L;
    sendFrame(r, q->getSrc(), Traffic::LlmCtl, 64, estDistanceTo(q->getSrc()), llmBusyUntil_ - simTime());
}

void RsuApp::pushModel()
{
    if (cbr_ > pushCbr_) return;
    int ver = currentVersion();
    int nChunks = std::max(1, (int) std::ceil(adapterBytes_ / chunkBytes_));
    // round-robin over fresh, in-range vehicles that have a base model but an old adapter
    std::vector<int> cand;
    for (auto& kv : nbrs_) {
        const Neighbor& n = kv.second;
        if (n.isRsu || beaconAge(n) > maxBeaconAge_) continue;
        if (n.modelVersion < 1 || n.modelVersion >= ver) continue;
        if (estDistanceTo(n.addr) > est_.commRange * 0.8) continue;
        cand.push_back(n.addr);
    }
    if (cand.empty()) return;
    int pick = cand.front();
    for (int a : cand)
        if (a > lastServed_) {
            pick = a;
            break;
        }
    lastServed_ = pick;
    const Neighbor& n = nbrs_[pick];
    auto& s = sent_[pick];
    if (s.first != ver) s = {ver, 0};
    int confirmed = (n.modelRecvVersion == ver) ? n.modelRecvChunks : 0;
    if (s.second > confirmed + 4 * chunksPerPush_ || s.second < confirmed) s.second = confirmed;  // resume
    double d = estDistanceTo(pick);
    for (int i = 0; i < chunksPerPush_ && s.second < nChunks; ++i, ++s.second) {
        auto* m = new LoModelChunk("modelChunk");
        m->setSrc(myId);
        m->setVersion(ver);
        m->setSeq(s.second);
        m->setNChunks(nChunks);
        m->setFederated(false);
        sendFrame(m, pick, Traffic::Model, std::min(chunkBytes_, adapterBytes_ - s.second * chunkBytes_), d);
    }
}

void RsuApp::onUnicastFailed(veins::BaseFrame1609_4* frame)
{
    if (auto* m = dynamic_cast<LoModelChunk*>(frame)) {
        auto it = sent_.find((int) m->getRecipientAddress());
        if (it != sent_.end() && it->second.first == m->getVersion()) it->second.second = std::min(it->second.second, m->getSeq());
    }
}

}  // namespace liteoff
