#include "VehicleApp.h"

#include <cstdint>
#include <sstream>

using namespace omnetpp;

namespace liteoff {

Define_Module(VehicleApp);

static long ctxId(cMessage* m) { return (long) (intptr_t) m->getContextPointer(); }

static cMessage* mkEvt(const char* name, int kind, long id)
{
    auto* m = new cMessage(name, kind);
    m->setContextPointer((void*) (intptr_t) id);
    return m;
}

VehicleApp::~VehicleApp()
{
    for (auto& kv : pend_) cancelEvents(kv.second);
    cancelAndDelete(genEvt_);
    cancelAndDelete(fedEvt_);
}

Vec2 VehicleApp::truePos() const
{
    return {curPosition.x, curPosition.y};
}

Vec2 VehicleApp::trueVel() const
{
    return {curSpeed.x, curSpeed.y};
}

static uint64_t fnv1a(const std::string& s)
{
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

void VehicleApp::initialize(int stage)
{
    LiteOffNodeApp::initialize(stage);
    if (stage != 0) return;
    cModule* w = world;
    {
        std::string seed = getEnvir()->getConfigEx()->getVariable("seedset");
        std::string ext = mobility ? mobility->getExternalId() : std::to_string(getId());
        wrng_.seed(fnv1a(seed + "/" + ext));
        if (cpuHz_ <= 0) cpuHz_ = wuni(par("cpuHzMin").doubleValue(), par("cpuHzMax").doubleValue());
    }
    scheme_ = w->par("scheme").stdstringValue();
    mode_ = w->par("deploymentMode").stdstringValue();
    families_ = w->par("contextFamilies").stdstringValue();
    llmModel_ = w->par("llmModel").stdstringValue();
    latKey_ = w->par("llmLatencyKey").stdstringValue();
    if (latKey_.empty()) latKey_ = llmModel_;
    rsuModel_ = w->par("rsuLlmModel").stdstringValue();
    K_ = w->par("K").intValue();
    gp_.enabled = w->par("gateEnabled").boolValue();
    gp_.useCache = w->par("cacheEnabled").boolValue();
    gp_.tinyCycles = w->par("tinyCycles").doubleValue();
    rw_.localMargin = w->par("localMargin").doubleValue();
    validatorOn_ = w->par("validatorEnabled").boolValue();
    ignoreLlmLatency_ = w->par("ignoreLlmLatency").boolValue();
    llmOnCpu_ = w->par("llmOnCpu").boolValue();
    reoffloadOn_ = w->par("reoffloadEnabled").boolValue();
    cloudOn_ = w->par("cloudEnabled").boolValue();
    escalateCritical_ = w->par("rsuEscalateCritical").boolValue();
    modelMgmt_ = w->par("modelManagement").boolValue();
    confThreshold_ = w->par("confThreshold").doubleValue();
    llmSpeedup_ = w->par("vehicleLlmSpeedup").doubleValue();
    llmMaxFrac_ = w->par("llmMaxBudgetFraction").doubleValue();
    lateGrace_ = w->par("lateGrace").doubleValue();
    watchdogFactor_ = w->par("watchdogFactor").doubleValue();
    minWatchdog_ = w->par("minWatchdog").doubleValue();
    cloudHz_ = w->par("cloudHz").doubleValue();
    cloudDelay_ = w->par("cloudDelay").doubleValue();
    exploreOracle_ = w->par("exploreOracleProb").doubleValue();
    exploreRandom_ = w->par("exploreRandomProb").doubleValue();
    queryBytesFactor_ = w->par("queryBytesFactor").doubleValue();
    fedPeriod_ = w->par("fedPeriod").doubleValue();
    fedModelBytes_ = w->par("fedModelBytes").doubleValue();
    rlMode_ = w->par("rlMode").stdstringValue();
    cache_ = DecisionCache(w->par("cacheTtl").doubleValue());
    taskRate_ = par("taskRate").doubleValue();
    taskStop_ = par("taskStopTime").doubleValue();
    energyBudget_ = par("energyBudgetJ").doubleValue();
    routeInBeacons_ = par("routeInBeacons").boolValue();
    uplinkBurst_ = par("uplinkBurst").intValue();
    uplinkPacing_ = par("uplinkPacing").doubleValue();
    richContext_ = true;
    chargeEnergy_ = true;
    modelVersion_ = (wuni(0, 1) < w->par("initialModelProb").doubleValue()) ? 1 : 0;
    selfish_ = wuni(0, 1) < par("selfishProb").doubleValue();

    genEvt_ = new cMessage("genTask", GEN_TASK_EVT);
    if (taskRate_ > 0) scheduleAt(simTime() + wexp(1.0 / taskRate_), genEvt_);
    if (scheme_ == "b4") {
        fedEvt_ = new cMessage("fedUpload", FED_EVT);
        scheduleAt(simTime() + uniform(0, fedPeriod_), fedEvt_);
    }
}

// ------------------------------------------------------------------ tasks
Task VehicleApp::generateTask()
{
    Task t;
    t.id = ((long) myId << 24) + (++taskCounter_);
    t.origin = myId;
    t.created = simTime().dbl();
    double u = wuni(0, 1);
    double pS = par("pSafety").doubleValue(), pP = par("pPerception").doubleValue();
    if (u < pS) {
        t.cls = TaskClass::Safety;
        t.bytes = wuni(par("safetyBytesMin").doubleValue(), par("safetyBytesMax").doubleValue());
        t.cycles = wuni(par("safetyCyclesMin").doubleValue(), par("safetyCyclesMax").doubleValue());
        t.tmax = wuni(par("safetyDeadlineMin").doubleValue(), par("safetyDeadlineMax").doubleValue());
        t.resultBytes = par("safetyResultBytes").doubleValue();
        t.prio = 2;
        t.splittable = false;
    }
    else if (u < pS + pP) {
        t.cls = TaskClass::Perception;
        t.bytes = wuni(par("percBytesMin").doubleValue(), par("percBytesMax").doubleValue());
        t.cycles = wuni(par("percCyclesMin").doubleValue(), par("percCyclesMax").doubleValue());
        t.tmax = wuni(par("percDeadlineMin").doubleValue(), par("percDeadlineMax").doubleValue());
        t.resultBytes = par("percResultBytes").doubleValue();
        t.prio = 1;
        t.splittable = true;
    }
    else {
        t.cls = TaskClass::Infotainment;
        t.bytes = wuni(par("infoBytesMin").doubleValue(), par("infoBytesMax").doubleValue());
        t.cycles = wuni(par("infoCyclesMin").doubleValue(), par("infoCyclesMax").doubleValue());
        t.tmax = wuni(par("infoDeadlineMin").doubleValue(), par("infoDeadlineMax").doubleValue());
        t.resultBytes = par("infoResultBytes").doubleValue();
        t.prio = 0;
        t.splittable = true;
    }
    t.tmax *= par("deadlineScale").doubleValue();
    return t;
}

void VehicleApp::handleSelfMsg(cMessage* msg)
{
    switch (msg->getKind()) {
    case GEN_TASK_EVT: {
        if (simTime().dbl() < taskStop_) {
            Task t = generateTask();
            Pending& p = pend_[t.id];
            p.task = t;
            p.rec.task = t;
            p.finalEvt = mkEvt("final", FINAL_EVT, t.id);
            scheduleAt(t.deadline() + lateGrace_, p.finalEvt);
            startDecision(p, false);
            scheduleAt(simTime() + wexp(1.0 / taskRate_), genEvt_);
        }
        return;
    }
    case DECISION_EVT: {
        auto it = pend_.find(ctxId(msg));
        if (it == pend_.end()) return;
        Pending& p = it->second;
        delete p.decisionEvt;
        p.decisionEvt = nullptr;
        act(p);
        return;
    }
    case WATCHDOG_EVT: {
        auto it = pend_.find(ctxId(msg));
        if (it == pend_.end()) return;
        delete it->second.watchdog;
        it->second.watchdog = nullptr;
        remoteFailed(it->second, "watchdog");
        return;
    }
    case UPLINK_EVT: {
        auto it = pend_.find(ctxId(msg));
        if (it == pend_.end()) return;
        pumpUplink(it->second);
        return;
    }
    case FINAL_EVT: {
        auto it = pend_.find(ctxId(msg));
        if (it == pend_.end()) return;
        delete it->second.finalEvt;
        it->second.finalEvt = nullptr;
        resolve(it->second, TaskStatus::Dropped);
        return;
    }
    case QUERY_TIMEOUT_EVT: {
        auto it = pend_.find(ctxId(msg));
        if (it == pend_.end()) return;
        Pending& p = it->second;
        delete p.queryTimeout;
        p.queryTimeout = nullptr;
        queryToTask_.erase(p.queryId);
        p.queryId = 0;
        fallback(p, p.actCtx, "rsu_timeout", 0);
        return;
    }
    case FED_EVT: {
        // B4: periodic upload of the local model update to the nearest RSU
        int rsu = -1;
        double best = 1e18;
        for (auto& kv : nbrs_) {
            if (!kv.second.isRsu || beaconAge(kv.second) > maxBeaconAge_) continue;
            double d = estDistanceTo(kv.first);
            if (d < best && d < est_.commRange) {
                best = d;
                rsu = kv.first;
            }
        }
        if (rsu >= 0) {
            int n = (int) std::ceil(fedModelBytes_ / chunkBytes_);
            for (int i = 0; i < n; ++i) {
                auto* m = new LoModelChunk("fedUp");
                m->setSrc(myId);
                m->setFederated(true);
                m->setSeq(i);
                m->setNChunks(n);
                sendFrame(m, rsu, Traffic::Fed, std::min(chunkBytes_, fedModelBytes_ - i * chunkBytes_), best);
            }
        }
        scheduleAt(simTime() + fedPeriod_, fedEvt_);
        return;
    }
    default:
        LiteOffNodeApp::handleSelfMsg(msg);
    }
}

// ------------------------------------------------------------------ context
std::string VehicleApp::nextEdges()
{
    if (!traciVehicle || !routeInBeacons_) return "";
    std::string road = mobility->getRoadId();
    if (road.empty() || road[0] == ':') road = roadId_;
    else roadId_ = road;
    auto route = traciVehicle->getPlannedRoadIds();
    std::ostringstream o;
    o << road;
    bool after = false;
    int n = 0;
    for (auto& e : route) {
        if (after && n < 3) {
            o << ' ' << e;
            ++n;
        }
        if (e == road) after = true;
    }
    return o.str();
}

void VehicleApp::fillBeacon(LoBeacon* b)
{
    LiteOffNodeApp::fillBeacon(b);
    b->setEnergy(std::max(0.0, 1.0 - energyUsedJ_ / energyBudget_));
    b->setModelVersion(modelVersion_);
    b->setModelRecvVersion(modelRecvVersion_);
    b->setModelRecvChunks(modelRecvChunks_);
    b->setNextEdges(nextEdges().c_str());
}

Candidate VehicleApp::makeCandidate(int addr, bool rsu, Vec2 pos, Vec2 vel, double cpu, double backlog, double load, int queueLen, double energy)
{
    Vec2 me = truePos(), vme = trueVel();
    Candidate c;
    c.nodeId = addr;
    c.kind = rsu ? NodeKind::Rsu : NodeKind::Vehicle;
    double dx = pos.x - me.x, dy = pos.y - me.y, dvx = vel.x - vme.x, dvy = vel.y - vme.y;
    c.dist = std::hypot(dx, dy);
    c.relSpeed = std::hypot(dvx, dvy);
    c.closing = c.dist > 1e-6 ? (dx * dvx + dy * dvy) / c.dist : 0;
    c.linkLifetime = linkLifetime(me, vme, pos, vel, est_.commRange);
    c.rateBps = rateForMcs(mcsTo(addr, c.dist));
    c.cpuHz = cpu;
    c.backlogSec = backlog;
    c.load = load;
    c.queueLen = queueLen;
    c.energy = energy;
    c.predDist2s = std::hypot(dx + 2 * dvx, dy + 2 * dvy);
    auto t = trust_.find(addr);
    c.trust = t != trust_.end() ? t->second : (rsu ? 0.95 : 0.8);
    auto to = timeouts_.find(addr);
    c.timeouts = to != timeouts_.end() ? (int) std::lround(to->second) : 0;
    return c;
}

Context VehicleApp::buildContext(const Task& t, const std::set<int>& exclude, bool rich, int K)
{
    Context ctx;
    ctx.now = simTime().dbl();
    ctx.task = t;
    SelfState& s = ctx.self;
    s.cpuHz = cpuHz_;
    s.backlogSec = backlogSec();
    s.queueLen = queueLen();
    s.load = load_;
    s.energy = std::max(0.0, 1.0 - energyUsedJ_ / energyBudget_);
    Vec2 v = trueVel();
    s.speed = std::hypot(v.x, v.y);
    s.cbr = cbr_;
    s.roadType = world->par("roadType").stdstringValue();

    std::string myEdges = rich ? nextEdges() : "";
    std::vector<Candidate> all;
    int fresh = 0;
    for (auto& kv : nbrs_) {
        const Neighbor& n = kv.second;
        double age = beaconAge(n);
        if (age > maxBeaconAge_) continue;
        ++fresh;
        if (exclude.count(n.addr)) continue;
        Vec2 pos{n.pos.x + n.vel.x * age, n.pos.y + n.vel.y * age};
        Candidate c = makeCandidate(n.addr, n.isRsu, pos, n.vel, n.cpuHz, std::max(0.0, n.backlogSec - age), n.load, n.queueLen, n.energy);
        c.loss = n.loss;
        c.rssi = n.rssi;
        c.beaconAge = age;
        if (rich && !n.isRsu && !n.nextEdges.empty()) {
            std::istringstream a(myEdges);
            std::string e;
            std::set<std::string> mine;
            while (a >> e) mine.insert(e);
            std::istringstream b(n.nextEdges);
            while (b >> e)
                if (mine.count(e)) c.routeOverlap = true;
        }
        all.push_back(c);
    }
    s.neighbors = fresh;
    double r = est_.commRange / 1000.0;
    s.density = fresh / (M_PI * r * r);
    // RSU knowledge (static infrastructure map) and coverage
    double rsuD = -1, cov = 0;
    Vec2 me = truePos();
    for (auto& kv : world->nodes()) {
        if (!kv.second->isRsu()) continue;
        Vec2 p = kv.second->truePos();
        double d = std::hypot(p.x - me.x, p.y - me.y);
        if (rsuD < 0 || d < rsuD) rsuD = d;
        cov = std::max(cov, linkLifetime(me, v, p, {0, 0}, est_.commRange));
    }
    s.rsuDist = rsuD;
    s.coverageLeft = cov;
    if (rich && traciVehicle) {
        std::string lane = traciVehicle->getLaneId();
        if (!lane.empty() && lane[0] == ':') s.junctionIn = 0;
        else if (!lane.empty()) {
            double len = traci->lane(lane).getLength();
            double pos = traciVehicle->getLanePosition();
            s.junctionIn = std::min(99.0, (len - pos) / std::max(0.5, s.speed));
        }
    }
    // cloud via the RSU with the longest remaining link
    if (cloudOn_) {
        const Candidate* bestRsu = nullptr;
        for (auto& c : all)
            if (c.kind == NodeKind::Rsu && (!bestRsu || c.linkLifetime > bestRsu->linkLifetime)) bestRsu = &c;
        if (bestRsu) {
            Candidate cl = *bestRsu;
            cl.kind = NodeKind::Cloud;
            cl.cpuHz = cloudHz_;
            cl.backlogSec = 0;
            cl.load = 0;
            cl.queueLen = 0;
            cl.trust = 0.99;
            cl.backhaulDelay = cloudDelay_;
            all.push_back(cl);
        }
    }
    ctx.cands = filterCandidates(all, t, cbr_, K > 0 ? K : K_, rw_, est_);
    return ctx;
}

Context VehicleApp::trueContext(const Context& ctx)
{
    Context tc = ctx;
    tc.self.backlogSec = backlogSec();
    for (auto& c : tc.cands) {
        LiteOffNodeApp* n = world->node(c.nodeId);
        if (!n) {
            c.linkLifetime = 0;
            continue;
        }
        Candidate t = makeCandidate(c.nodeId, n->isRsu(), n->truePos(), n->trueVel(), n->cpuHz(), n->backlogSec(), c.load, n->queueLen(), c.energy);
        c.dist = t.dist;
        c.linkLifetime = t.linkLifetime;
        c.rateBps = t.rateBps;
        if (c.kind != NodeKind::Cloud) {
            c.cpuHz = t.cpuHz;
            c.backlogSec = t.backlogSec;
        }
    }
    return tc;
}

// ------------------------------------------------------------------ decisions
void VehicleApp::startDecision(Pending& p, bool reoffload)
{
    Task t = p.task;
    if (reoffload) {
        // re-decide for the remote fraction only, with the remaining time budget
        t.bytes *= p.remoteFrac;
        t.cycles *= p.remoteFrac;
        t.resultBytes *= p.remoteFrac;
        t.splittable = false;
    }
    p.decisionStart = simTime();
    bool needRich = scheme_ == "liteoff" || scheme_ == "b2" || scheme_ == "b3" || world->par("datasetFile").stdstringValue() != "";
    Context ctx = buildContext(t, p.failedTargets, needRich);
    p.actCtx = ctx;
    if (!reoffload) {
        Context tc = trueContext(ctx);
        Decision o = oraclePolicy(tc, est_, true);
        p.rec.oracleSlot = o.slot;
        p.rec.oracleSplit = o.split;
        if (world->par("datasetFile").stdstringValue() != "") p.rec.ctxJson = contextToJson(ctx, families_, myId);
    }
    if (reoffload) {
        Decision d = rulePolicy(ctx, rw_, est_);
        d.path = "reoffload";
        d.split = 0;
        applyAfter(p, ctx, d, 0);
        return;
    }
    if (scheme_ == "local") {
        Decision d;
        d.path = "fixed";
        applyAfter(p, ctx, d, 0);
    }
    else if (scheme_ == "rsu") {
        Decision d;
        d.path = "fixed";
        double best = -1;
        for (auto& c : ctx.cands)
            if (c.kind == NodeKind::Rsu && c.linkLifetime > best) {
                best = c.linkLifetime;
                d.slot = c.slot;
            }
        applyAfter(p, ctx, d, 0);
    }
    else if (scheme_ == "random") {
        Decision d;
        d.path = "fixed";
        d.slot = intuniform(0, (int) ctx.cands.size());
        applyAfter(p, ctx, d, 0);
    }
    else if (scheme_ == "rule") {
        applyAfter(p, ctx, rulePolicy(ctx, rw_, est_), 0);
    }
    else if (scheme_ == "basic-rule") {
        applyAfter(p, ctx, basicRulePolicy(ctx, est_), 0);
    }
    else if (scheme_ == "oracle") {
        // all reachable neighbours (no top-K cap), true current state
        Context full = trueContext(buildContext(t, p.failedTargets, false, 1000));
        applyAfter(p, full, oraclePolicy(full, est_, true), 0);
    }
    else if (scheme_ == "explore") {
        double u = uniform(0, 1);
        Decision d;
        if (u < exploreOracle_) d = oraclePolicy(trueContext(ctx), est_, true);
        else if (u < exploreOracle_ + exploreRandom_) d.slot = intuniform(0, (int) ctx.cands.size());
        else d = rulePolicy(ctx, rw_, est_);
        d.path = "explore";
        applyAfter(p, ctx, d, 0);
    }
    else if (scheme_ == "liteoff") {
        decideLiteOff(p, ctx);
    }
    else if (scheme_ == "b1" || scheme_ == "b4" || scheme_ == "b5") {
        decideRl(p, ctx);
    }
    else if (scheme_ == "b2") {
        decideViaRsu(p, ctx, rsuModel_, families_, "b2");
    }
    else if (scheme_ == "b3") {
        decideViaRsu(p, ctx, rsuModel_, families_ + ",pred", "b3");
    }
    else {
        throw cRuntimeError("unknown scheme '%s'", scheme_.c_str());
    }
}

std::string VehicleApp::sidecarDecide(const std::string& scheme, const std::string& model, const Context& ctx, const std::string& families,
                                      std::map<std::string, std::string>& reply, double* wallMs)
{
    std::ostringstream req;
    req << "{\"op\":\"decide\",\"scheme\":\"" << scheme << "\",\"model\":\"" << jsonEscape(model) << "\",\"vid\":" << myId << ",\"tid\":" << ctx.task.id
        << ",\"region\":" << nearestRsuRegion() << ",\"mode\":\"" << rlMode_ << "\",\"ctx\":" << contextToJson(ctx, families, myId) << "}";
    std::string line = world->sidecar().request(req.str(), wallMs);
    bool ok = false;
    reply = parseFlatJson(line, &ok);
    if (!ok) reply.clear();
    return line;
}

void VehicleApp::chargeInference(double seconds, const std::string& model)
{
    double w = world->latency().has(model) ? world->latency().entry(model).powerW : 10.0;
    double j = w * seconds;
    energyUsedJ_ += j;
    world->countEnergy(0, 0, j);
}

void VehicleApp::decideLiteOff(Pending& p, const Context& ctx)
{
    double budget = ctx.budget();
    bool localFeasible = estimateLocal(ctx.task, ctx.self, 1.0) <= rw_.localMargin * budget;
    std::optional<Decision> hit;
    if (gp_.enabled && gp_.useCache && !localFeasible) hit = cache_.lookup(ctx, ctx.now);
    GatePath g = liteoff::gate(ctx, gp_, localFeasible, hit.has_value());
    if (g == GatePath::Local) {
        Decision d;
        d.path = "local";
        applyAfter(p, ctx, d, 0);
        return;
    }
    if (g == GatePath::Rule) {
        applyAfter(p, ctx, rulePolicy(ctx, rw_, est_), 0);
        return;
    }
    if (g == GatePath::Cache) {
        applyAfter(p, ctx, *hit, 0);
        return;
    }
    // ---- LLM path
    if (mode_ == "rsu") {
        decideViaRsu(p, ctx, rsuModel_, families_, "liteoff");
        return;
    }
    if (modelMgmt_ && modelVersion_ == 0) {
        world->countFallback("no_model");
        if (mode_ == "hybrid") decideViaRsu(p, ctx, rsuModel_, families_, "liteoff");
        else fallback(p, ctx, "", 0);
        return;
    }
    if (mode_ == "hybrid" && escalateCritical_ && ctx.task.prio == 2) {
        decideViaRsu(p, ctx, rsuModel_, families_, "liteoff");
        return;
    }
    std::map<std::string, std::string> rep;
    double wall = 0;
    std::string raw = sidecarDecide("liteoff", llmModel_, ctx, families_, rep, &wall);
    if (rep.empty() || rep.count("error")) {
        EV_WARN << "sidecar error: " << raw << endl;
        fallback(p, ctx, "sidecar_error", 0);
        return;
    }
    int nIn = std::stoi(rep["n_in"]), nOut = std::stoi(rep["n_out"]);
    double L = ignoreLlmLatency_ ? 0.0 : world->latency().latency(latKey_, nIn, nOut, llmSpeedup_);
    simtime_t now = simTime();
    double total;
    if (llmOnCpu_) {
        double wait = cpuBusy_ ? (currentEnd_ - now).dbl() : 0.0;
        total = wait + L;
        Job j;
        j.type = Job::Inference;
        j.cycles = L * cpuHz_;
        j.taskId = p.task.id;
        if (L > 0) enqueueJob(j, true);
    }
    else {
        simtime_t start = std::max(now, accelBusyUntil_);
        total = (start - now).dbl() + L;
        accelBusyUntil_ = start + L;
    }
    double maxL = llmMaxFrac_ * budget;
    world->countLlmCall(L, wall);
    if (total > maxL) {
        if (!llmOnCpu_) accelBusyUntil_ = now + maxL;
        chargeInference(std::min(L, maxL), latKey_);
        fallback(p, ctx, "llm_timeout", std::max(0.0, maxL));
        return;
    }
    chargeInference(L, latKey_);
    ParsedAnswer a = parseAnswer(rep.count("raw") ? rep["raw"] : "");
    if (rep.count("conf") && rep["conf"] != "null") a.conf = std::stod(rep["conf"]);
    if (!validatorOn_) {
        Decision d;
        d.path = "llm";
        if (!a.ok || a.slot < 0 || a.slot > (int) ctx.cands.size()) {
            world->countFallback("malformed_unvalidated");
            d.slot = 0;  // raw output unusable: execute locally
        }
        else {
            d.slot = a.slot;
            d.split = a.split;
        }
        d.conf = a.conf;
        applyAfter(p, ctx, d, total);
        return;
    }
    ValidationResult v = validate(a, ctx, est_, confThreshold_);
    if (v == ValidationResult::Ok) {
        Decision d;
        d.path = "llm";
        d.slot = a.slot;
        d.split = ctx.task.splittable ? a.split : 0;
        d.conf = a.conf;
        if (gp_.useCache) cache_.insert(ctx, d, ctx.now);
        applyAfter(p, ctx, d, total);
        return;
    }
    if (v == ValidationResult::LowConfidence && mode_ == "hybrid") {
        world->countFallback("low_confidence_escalated");
        // escalate after the on-board inference finished
        Pending* pp = &p;
        decideViaRsu(*pp, ctx, rsuModel_, families_, "liteoff");
        return;
    }
    fallback(p, ctx, validationName(v), total);
}

void VehicleApp::decideRl(Pending& p, const Context& ctx)
{
    std::map<std::string, std::string> rep;
    double wall = 0;
    std::string raw = sidecarDecide(scheme_, "", ctx, families_, rep, &wall);
    if (rep.empty() || rep.count("error")) {
        fallback(p, ctx, "sidecar_error", 0);
        return;
    }
    p.rlDecision = true;
    double L = world->latency().has("rl-mlp") ? world->latency().latency("rl-mlp", 0, 0, 1.0) : 0.001;
    Decision d;
    d.path = "rl";
    d.slot = std::stoi(rep["slot"]);
    if (d.slot < 0 || d.slot > (int) ctx.cands.size()) d.slot = 0;
    if (rep.count("split") && ctx.task.splittable && d.slot > 0) d.split = std::clamp(std::stoi(rep["split"]), 0, 100);  // B1 split head
    chargeInference(L, "rl-mlp");
    applyAfter(p, ctx, d, L);
}

int VehicleApp::nearestRsuRegion()
{
    Vec2 me = truePos();
    LiteOffNodeApp* r = world->nearestRsu(me.x, me.y);
    return r ? r->address() : -1;
}

void VehicleApp::decideViaRsu(Pending& p, const Context& ctx, const std::string& model, const std::string& families, const std::string& scheme)
{
    // pick reachable RSU (longest link) among fresh RSU neighbours
    int rsu = -1;
    double bestLL = -1, dist = 0;
    for (auto& kv : nbrs_) {
        const Neighbor& n = kv.second;
        if (!n.isRsu || beaconAge(n) > maxBeaconAge_) continue;
        Vec2 me = truePos();
        double ll = linkLifetime(me, trueVel(), n.pos, {0, 0}, est_.commRange);
        if (ll > bestLL) {
            bestLL = ll;
            rsu = n.addr;
            dist = std::hypot(n.pos.x - me.x, n.pos.y - me.y);
        }
    }
    if (rsu < 0 || bestLL <= 0) {
        fallback(p, ctx, "no_rsu", 0);
        return;
    }
    std::ostringstream req;
    std::string cj = contextToJson(ctx, families, myId);
    req << "{\"op\":\"decide\",\"scheme\":\"" << scheme << "\",\"model\":\"" << jsonEscape(model) << "\",\"vid\":" << myId << ",\"tid\":" << ctx.task.id
        << ",\"region\":" << rsu << ",\"mode\":\"" << rlMode_ << "\",\"ctx\":" << cj << "}";
    long qid = world->storeQuery(req.str());
    p.queryId = qid;
    p.actCtx = ctx;
    queryToTask_[qid] = p.task.id;
    auto* q = new LoLlmQuery("llmQuery");
    q->setSrc(myId);
    q->setQueryId(qid);
    sendFrame(q, rsu, Traffic::LlmCtl, std::max(200.0, cj.size() * queryBytesFactor_), dist);
    p.queryTimeout = mkEvt("queryTimeout", QUERY_TIMEOUT_EVT, p.task.id);
    scheduleAt(simTime() + std::max(0.005, llmMaxFrac_ * ctx.budget()), p.queryTimeout);
}

void VehicleApp::onLlmReply(LoLlmReply* r)
{
    auto qt = queryToTask_.find(r->getQueryId());
    if (qt == queryToTask_.end()) return;
    long tid = qt->second;
    queryToTask_.erase(qt);
    auto it = pend_.find(tid);
    if (it == pend_.end()) return;
    Pending& p = it->second;
    if (p.queryTimeout) {
        cancelAndDelete(p.queryTimeout);
        p.queryTimeout = nullptr;
    }
    p.queryId = 0;
    // the context may have aged by the round trip; validate against the original candidate list
    Context ctx = p.actCtx;
    ParsedAnswer a;
    a.ok = r->getOk();
    a.slot = r->getSlot();
    a.split = r->getSplit();
    a.conf = -1;  // RSU model is the final authority: no confidence gate
    ValidationResult v = a.ok ? (a.slot >= 0 && a.slot <= (int) ctx.cands.size() ? ValidationResult::Ok : ValidationResult::NotCandidate) : ValidationResult::Malformed;
    if (v == ValidationResult::Ok && validatorOn_ && scheme_ == "liteoff") v = validate(a, ctx, est_, 0.0);
    if (v != ValidationResult::Ok) {
        fallback(p, ctx, std::string("rsu_") + validationName(v), 0);
        return;
    }
    Decision d;
    d.path = "rsu-llm";
    d.slot = a.slot;
    d.split = ctx.task.splittable ? a.split : 0;
    applyAfter(p, ctx, d, 0);
}

void VehicleApp::fallback(Pending& p, const Context& ctx, const std::string& reason, double delay)
{
    if (!reason.empty()) world->countFallback(reason);
    p.rec.fallbackReason = reason;
    Decision d = rulePolicy(ctx, rw_, est_);
    d.path = "fallback";
    d.fromFallback = true;
    applyAfter(p, ctx, d, delay);
}

void VehicleApp::applyAfter(Pending& p, const Context& ctx, Decision d, double delay)
{
    p.actCtx = ctx;
    p.pendingDecision = d;
    if (delay <= 0) {
        act(p);
        return;
    }
    p.decisionEvt = mkEvt("decision", DECISION_EVT, p.task.id);
    scheduleAt(simTime() + delay, p.decisionEvt);
}

// ------------------------------------------------------------------ execution
void VehicleApp::act(Pending& p)
{
    const Decision& d = p.pendingDecision;
    const Context& ctx = p.actCtx;
    double now = simTime().dbl();
    bool reoffload = p.decided;
    if (!reoffload) {
        p.decided = true;
        p.rec.path = d.path;
        p.rec.decisionDelay = now - p.task.created;
        p.rec.chosenSlot = d.slot;
        world->countPath(d.path);
        world->countDecision(now - p.decisionStart.dbl());
        // oracle agreement on the same candidate list (target identity)
        if (scheme_ == "oracle") p.rec.oracleAgree = true;
        else p.rec.oracleAgree = (d.slot == p.rec.oracleSlot);
    }
    int slot = (d.slot >= 0 && d.slot <= (int) ctx.cands.size()) ? d.slot : 0;
    double keepLocal = (!reoffload && slot > 0 && p.task.splittable) ? std::clamp(d.split, 0, 100) / 100.0 : 0.0;
    if (!reoffload) {
        p.rec.split = slot > 0 ? (int) std::lround(keepLocal * 100) : 0;
        p.rec.targetKind = slot == 0 ? "local" : kindName(ctx.cands[slot - 1].kind);
    }
    double frac = reoffload ? p.remoteFrac : 1.0;  // fraction of the task this decision covers
    if (slot == 0) {
        Job j;
        j.type = Job::LocalPart;
        j.taskId = p.task.id;
        j.cycles = p.task.cycles * frac;
        j.deadline = p.task.deadline() + lateGrace_;
        j.ref = reoffload ? 1 : 0;  // 1: executes the (failed) remote part locally
        if (reoffload) p.remoteNeeded = true;
        else p.localNeeded = true;
        enqueueJob(j);
        return;
    }
    const Candidate& c = ctx.cands[slot - 1];
    if (keepLocal > 0) {
        Job j;
        j.type = Job::LocalPart;
        j.taskId = p.task.id;
        j.cycles = p.task.cycles * keepLocal;
        j.deadline = p.task.deadline() + lateGrace_;
        p.localNeeded = true;
        enqueueJob(j);
    }
    p.remoteNeeded = true;
    p.remoteFrac = reoffload ? p.remoteFrac : 1.0 - keepLocal;
    sendTaskPart(p, c, p.remoteFrac);
}

void VehicleApp::sendTaskPart(Pending& p, const Candidate& c, double frac)
{
    p.target = c.nodeId;
    p.targetKind = c.kind;
    p.sendTime = simTime();
    p.rec.offloaded = true;
    double bytes = p.task.bytes * frac;
    p.upN = std::max(1, (int) std::ceil(bytes / chunkBytes_));
    p.upNext = 0;
    p.upBytes = bytes;
    p.upFrac = frac;
    p.upMcs = mcsTo(c.nodeId, c.dist);
    if (!p.uplinkEvt) p.uplinkEvt = mkEvt("uplink", UPLINK_EVT, p.task.id);
    else if (p.uplinkEvt->isScheduled()) cancelEvent(p.uplinkEvt);
    pumpUplink(p);
    Task t = p.task;
    t.bytes = bytes;
    t.cycles *= frac;
    t.resultBytes *= frac;
    double est = estimateRemote(t, c, 1.0, cbr_, est_);
    double wd = std::max(minWatchdog_, watchdogFactor_ * est);
    if (p.watchdog) cancelAndDelete(p.watchdog);
    p.watchdog = mkEvt("watchdog", WATCHDOG_EVT, p.task.id);
    scheduleAt(simTime() + std::min(wd, std::max(minWatchdog_, p.task.deadline() - simTime().dbl())), p.watchdog);
}

void VehicleApp::pumpUplink(Pending& p)
{
    // paced upload: a few chunks per airtime slot, so a broken link does not flush the whole task into the MAC
    for (int k = 0; k < uplinkBurst_ && p.upNext < p.upN; ++k, ++p.upNext) {
        int i = p.upNext;
        auto* m = new LoTaskChunk("taskChunk");
        m->setSrc(myId);
        m->setTaskId(p.task.id);
        m->setAttempt(p.attempt);
        m->setOrigin(myId);
        m->setSeq(i);
        m->setNChunks(p.upN);
        m->setPartBytes(p.upBytes);
        m->setCycles(p.task.cycles * p.upFrac);
        m->setResultBytes(p.task.resultBytes * p.upFrac);
        m->setCreated(p.task.created);
        m->setDeadline(p.task.deadline());
        m->setToCloud(p.targetKind == NodeKind::Cloud);
        sendFrame(m, p.target, Traffic::Task, std::min(chunkBytes_, p.upBytes - i * chunkBytes_), estDistanceTo(p.target));
    }
    if (p.upNext < p.upN) scheduleAt(simTime() + uplinkPacing_ * uplinkBurst_ * airtime(chunkBytes_, p.upMcs), p.uplinkEvt);
}

void VehicleApp::remoteFailed(Pending& p, const std::string& why)
{
    if (p.remoteDone || !p.remoteNeeded) return;
    if (p.uplinkEvt && p.uplinkEvt->isScheduled()) cancelEvent(p.uplinkEvt);
    p.upNext = p.upN;
    if (p.watchdog) {
        cancelAndDelete(p.watchdog);
        p.watchdog = nullptr;
    }
    updateTrust(p.target, false);
    timeouts_[p.target] += 1;
    double remaining = p.task.deadline() - simTime().dbl();
    if (reoffloadOn_ && !p.reoffloaded && remaining > 0) {
        p.reoffloaded = true;
        p.rec.reoffloaded = true;
        p.attempt++;
        p.failedTargets.insert(p.target);
        world->countReoffload();
        startDecision(p, true);
        return;
    }
    if (!p.reoffloaded) {
        // no re-offload: run the remote part locally
        p.reoffloaded = true;
        p.attempt++;
        Job j;
        j.type = Job::LocalPart;
        j.taskId = p.task.id;
        j.cycles = p.task.cycles * p.remoteFrac;
        j.deadline = p.task.deadline() + lateGrace_;
        j.ref = 1;
        enqueueJob(j);
    }
    // otherwise wait for the final timeout
    (void) why;
}

void VehicleApp::onUnicastFailed(veins::BaseFrame1609_4* frame)
{
    if (auto* c = dynamic_cast<LoTaskChunk*>(frame)) {
        auto it = pend_.find(c->getTaskId());
        if (it == pend_.end()) return;
        if (it->second.attempt != c->getAttempt()) return;
        remoteFailed(it->second, "mac");
    }
    else if (auto* q = dynamic_cast<LoLlmQuery*>(frame)) {
        auto qt = queryToTask_.find(q->getQueryId());
        if (qt == queryToTask_.end()) return;
        auto it = pend_.find(qt->second);
        queryToTask_.erase(qt);
        if (it == pend_.end()) return;
        Pending& p = it->second;
        if (p.queryTimeout) {
            cancelAndDelete(p.queryTimeout);
            p.queryTimeout = nullptr;
        }
        p.queryId = 0;
        fallback(p, p.actCtx, "rsu_unreachable", 0);
    }
}

void VehicleApp::onResult(LoResult* r)
{
    if (r->getOrigin() != myId) return;
    auto it = pend_.find(r->getTaskId());
    if (it == pend_.end()) return;
    Pending& p = it->second;
    if (p.remoteDone || !p.remoteNeeded) return;
    p.remoteDone = true;
    if (p.watchdog) {
        cancelAndDelete(p.watchdog);
        p.watchdog = nullptr;
    }
    if (p.uplinkEvt && p.uplinkEvt->isScheduled()) cancelEvent(p.uplinkEvt);
    updateTrust(p.target, true);
    double now = simTime().dbl();
    if (r->getRxDone() > 0) {
        p.rec.txDelay = (r->getRxDone() - p.sendTime).dbl();
        p.rec.queueDelay = (r->getExecStart() - r->getRxDone()).dbl();
        p.rec.execDelay = (r->getExecEnd() - r->getExecStart()).dbl();
        p.rec.resultDelay = now - r->getExecEnd().dbl();
    }
    checkComplete(p);
}

void VehicleApp::onJobDone(const Job& j)
{
    if (j.type == Job::Served) {
        LiteOffNodeApp::onJobDone(j);
        return;
    }
    if (j.type != Job::LocalPart) return;
    auto it = pend_.find(j.taskId);
    if (it == pend_.end()) return;
    Pending& p = it->second;
    if (j.ref == 1) {
        if (!p.remoteDone) {
            p.remoteDone = true;
            if (p.watchdog) {
                cancelAndDelete(p.watchdog);
                p.watchdog = nullptr;
            }
        }
    }
    else {
        p.localDone = true;
        if (!p.rec.offloaded) {
            p.rec.queueDelay = (j.started - j.enqueued).dbl();
            p.rec.execDelay = (simTime() - j.started).dbl();
        }
    }
    checkComplete(p);
}

void VehicleApp::onTaskReceived(int origin, long taskId, int attempt, const RxPartial& p)
{
    if (selfish_) return;  // E9: selfish/malicious node accepts the upload but never returns a result
    LiteOffNodeApp::onTaskReceived(origin, taskId, attempt, p);
}

void VehicleApp::checkComplete(Pending& p)
{
    if (!p.decided) return;
    if (p.localNeeded && !p.localDone) return;
    if (p.remoteNeeded && !p.remoteDone) return;
    resolve(p, simTime().dbl() <= p.task.deadline() + 1e-12 ? TaskStatus::InTime : TaskStatus::Late);
}

void VehicleApp::updateTrust(int addr, bool ok)
{
    if (addr < 0) return;
    auto it = trust_.find(addr);
    double t = it != trust_.end() ? it->second : 0.8;
    trust_[addr] = 0.9 * t + (ok ? 0.1 : 0.0);
    if (ok) timeouts_[addr] *= 0.5;
}

void VehicleApp::cancelEvents(Pending& p)
{
    for (cMessage** m : {&p.watchdog, &p.finalEvt, &p.decisionEvt, &p.queryTimeout, &p.uplinkEvt}) {
        if (*m) {
            cancelAndDelete(*m);
            *m = nullptr;
        }
    }
    if (p.queryId) queryToTask_.erase(p.queryId);
}

void VehicleApp::resolve(Pending& p, TaskStatus st)
{
    double e2e = simTime().dbl() - p.task.created;
    if (p.rec.offloaded && st != TaskStatus::Dropped) {
        // offload succeeded if the remote result came back
    }
    world->taskResolved(p.rec, st, e2e);
    if (p.rlDecision) {
        double r = st == TaskStatus::InTime ? 1.0 - 0.5 * e2e / p.task.tmax : (st == TaskStatus::Late ? -0.5 : -1.0);
        std::ostringstream o;
        o << "{\"op\":\"feedback\",\"scheme\":\"" << scheme_ << "\",\"vid\":" << myId << ",\"tid\":" << p.task.id << ",\"reward\":" << r
          << ",\"reoffloaded\":" << (p.rec.reoffloaded ? 1 : 0) << ",\"mode\":\"" << rlMode_ << "\",\"t\":" << simTime().dbl() << "}";
        world->sidecar().request(o.str());
    }
    cancelEvents(p);
    pend_.erase(p.task.id);
}

void VehicleApp::flushPending()
{
    std::vector<long> ids;
    for (auto& kv : pend_) ids.push_back(kv.first);
    for (long id : ids) {
        auto it = pend_.find(id);
        if (it != pend_.end()) resolve(it->second, TaskStatus::Dropped);
    }
}

// ------------------------------------------------------------------ model client
void VehicleApp::onModelChunk(LoModelChunk* m)
{
    if (m->getFederated()) return;  // B4 aggregated model broadcast (cost already counted)
    if (m->getVersion() <= modelVersion_) return;
    if (modelRecvVersion_ != m->getVersion()) {
        modelRecvVersion_ = m->getVersion();
        modelRecvChunks_ = 0;
    }
    if (m->getSeq() == modelRecvChunks_) ++modelRecvChunks_;
    if (modelRecvChunks_ >= m->getNChunks()) {
        modelVersion_ = m->getVersion();
        modelRecvChunks_ = 0;
        world->countModelUpdate();
    }
}

}  // namespace liteoff
