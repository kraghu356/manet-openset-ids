#include "ns3/aodv-module.h"
#include "ns3/aodvatk-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <map>
#include <array>
#include <cmath>
#include <set>
#include <set>
#include <set>
#include "ns3/integer.h"

using namespace ns3;
namespace ns3 { namespace aodvatk {
uint64_t GetBhDrops(); uint64_t GetBhSeen();
uint64_t CtrFwdSeen(uint32_t); uint64_t CtrFwdOk(uint32_t);
uint64_t CtrFwdDrop(uint32_t); uint64_t CtrRreqRecv(uint32_t);
uint64_t CtrNoRoute(uint32_t);
uint64_t CtrRrepSent(uint32_t);
const std::vector<uint32_t>& NbAddresses(uint32_t);
} }




struct Cfg {
  uint32_t nNodes = 50;
  double areaX = 700.0, areaY = 700.0;
  double simTime = 300.0;
  double minSpeed = 1.0, maxSpeed = 5.0, pause = 2.0;
  uint32_t nFlows = 10;
  double pktRate = 4.0;
  uint32_t pktSize = 512;
  std::string standard = "80211g";
  std::string dataMode = "ErpOfdmRate54Mbps";
  double txPowerDbm = 20.0;
  double plExp = 3.0, refLoss = 46.6777;
  bool enableHello = false;
  double routeTimeout = 10.0;
  uint32_t nMalicious = 0;
  double dropProb = 1.0;
  bool forgeRrep = true;
  double floodRate = 0.0;
  bool wormhole = false;
  uint32_t seed = 1;
  std::string outDir = "out";
};
static Cfg g;

static std::ofstream g_feat;
static std::map<uint32_t, std::array<uint64_t,6>> g_prev;
// Per-node RSSI statistics, gathered from the PHY monitor trace. A node can
// measure received signal strength; it cannot measure geometric distance.
struct RssiStat { double sum=0, mn=1e9, mx=-1e9; uint64_t n=0; };
static std::map<uint32_t, RssiStat> g_rssi;
static std::map<Mac48Address, uint32_t> g_macToNode;
static std::map<uint32_t, std::set<uint32_t>> g_heard;   // nodes heard on air
static void SnifferRx(std::string ctx, Ptr<const Packet> p, uint16_t freq,
                      WifiTxVector tx, MpduInfo mpdu, SignalNoiseDbm sn,
                      uint16_t staId) {
  size_t a=ctx.find("/NodeList/")+10; size_t b=ctx.find("/",a);
  uint32_t id=std::stoul(ctx.substr(a,b-a));
  auto& r=g_rssi[id];
  r.sum+=sn.signal; r.n++;
  if (sn.signal<r.mn) r.mn=sn.signal;
  if (sn.signal>r.mx) r.mx=sn.signal;
  WifiMacHeader hdr;
  Ptr<Packet> c=p->Copy();
  if (c->PeekHeader(hdr)) {
    auto it=g_macToNode.find(hdr.GetAddr2());
    if (it!=g_macToNode.end()) g_heard[id].insert(it->second);
  }
}
// Per-node RSSI statistics, gathered from the PHY monitor trace. A node can
// measure received signal strength; it cannot measure geometric distance.

// Per-node RSSI statistics, gathered from the PHY monitor trace. A node can
// measure received signal strength; it cannot measure geometric distance.

static void EmitWindow(NodeContainer nodes, uint32_t nMal, double win) {
  double t = Simulator::Now().GetSeconds();
  for (uint32_t i = 0; i < nodes.GetN(); ++i) {
    uint64_t a = aodvatk::CtrFwdSeen(i), b = aodvatk::CtrFwdOk(i);
    uint64_t c = aodvatk::CtrFwdDrop(i), d = aodvatk::CtrRreqRecv(i);
    uint64_t e = aodvatk::CtrNoRoute(i);
    uint64_t f = aodvatk::CtrRrepSent(i);
    auto& pv = g_prev[i];
    uint64_t ds=a-pv[0], dok=b-pv[1], ddr=c-pv[2], drq=d-pv[3], dnr=e-pv[4], drs=f-pv[5];
    pv = {a,b,c,d,e,f};
    // Topology features. A wormhole endpoint lists neighbours far beyond
    // radio range, which no honest node and no other attack produces.
    Ptr<MobilityModel> mm = nodes.Get(i)->GetObject<MobilityModel>();
    Vector myp = mm->GetPosition();
    const std::vector<uint32_t>& nbs = aodvatk::NbAddresses(i);
    double dsum = 0.0, dmax = 0.0;
    for (uint32_t nb : nbs) {
      if (nb >= nodes.GetN()) continue;
      Vector q = nodes.Get(nb)->GetObject<MobilityModel>()->GetPosition();
      double dd = CalculateDistance(myp, q);
      dsum += dd; if (dd > dmax) dmax = dd;
    }
    double dmean = nbs.empty() ? 0.0 : dsum / nbs.size();

    // Observable substitutes for geometric distance.
    // rssi_dist_max: farthest neighbour by RSSI-inverted path loss, using the
    //   same constants a node would be configured with.
    // nb_no_rssi: neighbours in the routing table never actually heard on the
    //   air. A wormhole partner is reachable through the tunnel but is not a
    //   radio neighbour, so this is non-zero only for tunnel endpoints.
    // nb_churn: neighbours entering or leaving the set since the last window.
    const RssiStat& rs = g_rssi[i];
    double rssiMin = (rs.n ? rs.mn : -100.0);
    double rssiDistMax = std::pow(10.0,
        (g.txPowerDbm - g.refLoss - rssiMin) / (10.0 * g.plExp));
    uint32_t noRssi = 0;
    for (uint32_t nb : nbs)
      if (g_heard[i].find(nb) == g_heard[i].end()) noRssi++;
    static std::map<uint32_t, std::set<uint32_t>> prevNb;
    std::set<uint32_t> cur(nbs.begin(), nbs.end());
    uint32_t churn = 0;
    for (uint32_t x : cur) if (!prevNb[i].count(x)) churn++;
    for (uint32_t x : prevNb[i]) if (!cur.count(x)) churn++;
    prevNb[i] = cur;


    Vector v = mm->GetVelocity();
    // Ratio excludes packets that could not be forwarded for lack of a
    // route: that is a routing failure, not misbehaviour, and honest nodes
    // experience it constantly under mobility.
    uint64_t den = dok + ddr;
    double fr = den ? (double)dok/den : 1.0;
    bool mal = (i >= nodes.GetN() - nMal);
    g_feat << t << "," << i << "," << ds << "," << dok << "," << ddr << ","
           << fr << "," << drq << "," << dnr << "," << drs << "," << nbs.size() << "," << dmean << "," << dmax << ","
           << rssiDistMax << "," << noRssi << "," << churn << "," << rssiMin << ","
           << std::sqrt(v.x*v.x+v.y*v.y) << ","
           << (mal?1:0) << "," << (mal?"BHA":"Normal") << "\n";
  }
  g_feat.flush();
  Simulator::Schedule(Seconds(win), &EmitWindow, nodes, nMal, win);
}
int main(int argc, char* argv[]) {
  CommandLine cmd(__FILE__);
  cmd.AddValue("nNodes", "nodes", g.nNodes);
  cmd.AddValue("areaX", "field width", g.areaX);
  cmd.AddValue("areaY", "field height", g.areaY);
  cmd.AddValue("simTime", "duration", g.simTime);
  cmd.AddValue("minSpeed", "min speed", g.minSpeed);
  cmd.AddValue("maxSpeed", "max speed", g.maxSpeed);
  cmd.AddValue("pause", "pause time", g.pause);
  cmd.AddValue("nFlows", "flows", g.nFlows);
  cmd.AddValue("pktRate", "pkt/s per flow", g.pktRate);
  cmd.AddValue("pktSize", "payload bytes", g.pktSize);
  cmd.AddValue("standard", "80211b|80211g", g.standard);
  cmd.AddValue("dataMode", "phy rate", g.dataMode);
  cmd.AddValue("txPower", "dBm", g.txPowerDbm);
  cmd.AddValue("plExp", "path loss exponent", g.plExp);
  cmd.AddValue("refLoss", "ref loss dB", g.refLoss);
  cmd.AddValue("hello", "aodv hello", g.enableHello);
  cmd.AddValue("routeTimeout", "active route timeout s", g.routeTimeout);
  cmd.AddValue("nMalicious", "number of black hole nodes", g.nMalicious);
  cmd.AddValue("wormhole", "pair the two attackers as a tunnel", g.wormhole);
  cmd.AddValue("floodRate", "bogus RREQs/s per attacker", g.floodRate);
  cmd.AddValue("forge", "forge RREPs (0=drop only)", g.forgeRrep);
  cmd.AddValue("dropProb", "drop probability for malicious nodes", g.dropProb);
  cmd.AddValue("seed", "rng seed", g.seed);
  cmd.AddValue("outDir", "output dir", g.outDir);
  cmd.Parse(argc, argv);
  RngSeedManager::SetSeed(g.seed);
  RngSeedManager::SetRun(1);
  mkdir(g.outDir.c_str(), 0755);
  NodeContainer nodes;
  nodes.Create(g.nNodes);
  WifiHelper wifi;
  if (g.standard == "80211b") wifi.SetStandard(WIFI_STANDARD_80211b);
  else wifi.SetStandard(WIFI_STANDARD_80211g);
  wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                               "DataMode", StringValue(g.dataMode),
                               "ControlMode", StringValue("ErpOfdmRate6Mbps"));
  YansWifiChannelHelper ch;
  ch.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
  ch.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                        "Exponent", DoubleValue(g.plExp),
                        "ReferenceLoss", DoubleValue(g.refLoss));
  YansWifiPhyHelper phy;
  phy.SetChannel(ch.Create());
  phy.Set("TxPowerStart", DoubleValue(g.txPowerDbm));
  phy.Set("TxPowerEnd", DoubleValue(g.txPowerDbm));
  WifiMacHelper mac;
  mac.SetType("ns3::AdhocWifiMac");
  NetDeviceContainer devices = wifi.Install(phy, mac, nodes);
  for (uint32_t i = 0; i < devices.GetN(); ++i) {
    Ptr<WifiNetDevice> wd = DynamicCast<WifiNetDevice>(devices.Get(i));
    if (wd) g_macToNode[Mac48Address::ConvertFrom(wd->GetAddress())] =
              devices.Get(i)->GetNode()->GetId();
  }
  Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/MonitorSnifferRx",
                  MakeCallback(&SnifferRx));

  for (uint32_t i = 0; i < devices.GetN(); ++i) {
    Ptr<WifiNetDevice> wd = DynamicCast<WifiNetDevice>(devices.Get(i));
    if (wd) g_macToNode[Mac48Address::ConvertFrom(wd->GetAddress())] =
              devices.Get(i)->GetNode()->GetId();
  }
  Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/MonitorSnifferRx",
                  MakeCallback(&SnifferRx));

  std::string xs = "ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(g.areaX) + "]";
  std::string ys = "ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(g.areaY) + "]";
  Ptr<RandomRectanglePositionAllocator> pa =
      CreateObject<RandomRectanglePositionAllocator>();
  pa->SetAttribute("X", StringValue(xs));
  pa->SetAttribute("Y", StringValue(ys));
  MobilityHelper mob;
  mob.SetPositionAllocator(pa);
  mob.SetMobilityModel("ns3::RandomWaypointMobilityModel",
      "Speed", StringValue("ns3::UniformRandomVariable[Min=" +
                std::to_string(g.minSpeed) + "|Max=" + std::to_string(g.maxSpeed) + "]"),
      "Pause", StringValue("ns3::ConstantRandomVariable[Constant=" +
                std::to_string(g.pause) + "]"),
      "PositionAllocator", PointerValue(pa));
  mob.Install(nodes);
  AodvHelper aodv;
  aodv.Set("EnableHello", BooleanValue(g.enableHello));
  aodv.Set("ActiveRouteTimeout", TimeValue(Seconds(g.routeTimeout)));
  // Malicious nodes are the LAST nMalicious ids, so honest node indices are
  // stable across runs with different attacker counts.
  AodvAtkHelper atk;
  atk.Set("EnableHello", BooleanValue(g.enableHello));
  atk.Set("ActiveRouteTimeout", TimeValue(Seconds(g.routeTimeout)));
  atk.Set("EnableBlackHole", BooleanValue(true));
  atk.Set("DropProb", DoubleValue(g.dropProb));
  atk.Set("ForgeRrep", BooleanValue(g.forgeRrep));
  atk.Set("FloodRate", DoubleValue(g.floodRate));
  NodeContainer honest, malicious;
  for (uint32_t i = 0; i < g.nNodes; ++i) {
    if (i >= g.nNodes - g.nMalicious) malicious.Add(nodes.Get(i));
    else honest.Add(nodes.Get(i));
  }
  // Every node runs the instrumented module so that per-node counters are
  // collected uniformly. Honest nodes simply have the attack disabled.
  AodvAtkHelper honestStack;
  honestStack.Set("EnableHello", BooleanValue(g.enableHello));
  honestStack.Set("ActiveRouteTimeout", TimeValue(Seconds(g.routeTimeout)));
  honestStack.Set("EnableBlackHole", BooleanValue(false));
  InternetStackHelper internet;
  internet.SetRoutingHelper(honestStack);
  internet.Install(honest);
  if (g.nMalicious > 0) {
    if (g.wormhole && g.nMalicious >= 2) {
      // Two endpoints only, each naming the other. Installed one at a time
      // because the partner id differs per node.
      uint32_t a = g.nNodes - g.nMalicious;
      uint32_t b = g.nNodes - 1;
      AodvAtkHelper wa = atk, wb = atk;
      wa.Set("WormPartner", IntegerValue((int32_t)b));
      wb.Set("WormPartner", IntegerValue((int32_t)a));
      InternetStackHelper sa; sa.SetRoutingHelper(wa); sa.Install(nodes.Get(a));
      InternetStackHelper sb; sb.SetRoutingHelper(wb); sb.Install(nodes.Get(b));
      NodeContainer rest;
      for (uint32_t i = a + 1; i < b; ++i) rest.Add(nodes.Get(i));
      if (rest.GetN() > 0) {
        InternetStackHelper sr; sr.SetRoutingHelper(atk); sr.Install(rest);
      }
    } else {
      InternetStackHelper atkStack;
      atkStack.SetRoutingHelper(atk);
      atkStack.Install(malicious);
    }
  }
  Ipv4AddressHelper addr;
  addr.SetBase("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer ifs = addr.Assign(devices);
  uint16_t port = 9;
  Ptr<UniformRandomVariable> rv = CreateObject<UniformRandomVariable>();
  uint32_t half = g.nNodes / 2;
  for (uint32_t f = 0; f < g.nFlows; ++f) {
    uint32_t src = f % half;
    uint32_t dst = half + (f % half);
    PacketSinkHelper sink("ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port + f));
    ApplicationContainer sa = sink.Install(nodes.Get(dst));
    sa.Start(Seconds(0.0)); sa.Stop(Seconds(g.simTime));
    OnOffHelper on("ns3::UdpSocketFactory",
        InetSocketAddress(ifs.GetAddress(dst), port + f));
    on.SetConstantRate(DataRate((uint64_t)(g.pktRate * g.pktSize * 8)), g.pktSize);
    ApplicationContainer app = on.Install(nodes.Get(src));
    app.Start(Seconds(1.0 + rv->GetValue(0.0, 5.0)));
    app.Stop(Seconds(g.simTime));
  }
  g_feat.open(g.outDir + "/features.csv");
  g_feat << "time_s,node_id,fwd_seen,fwd_ok,fwd_drop,fwd_ratio,rreq_recv,no_route,rreq_sent,nb_count,nb_mean_dist,nb_max_dist,rssi_dist_max,nb_no_rssi,nb_churn,rssi_min,speed,is_malicious,label\n";
  Simulator::Schedule(Seconds(10.0), &EmitWindow, nodes, g.nMalicious, 10.0);
  FlowMonitorHelper fmh;
  Ptr<FlowMonitor> mon = fmh.InstallAll();
  Simulator::Stop(Seconds(g.simTime));
  Simulator::Run();
  mon->CheckForLostPackets();
  FlowMonitor::FlowStatsContainer st = mon->GetFlowStats();
  uint64_t tx = 0, rx = 0; double dsum = 0.0; uint64_t rxd = 0;
  uint64_t fwd = 0;
  for (auto& kv : st) {
    tx += kv.second.txPackets; rx += kv.second.rxPackets;
    dsum += kv.second.delaySum.GetSeconds(); rxd += kv.second.rxPackets;
    fwd += kv.second.timesForwarded;
  }
  double pdr = tx ? (double)rx / tx : 0.0;
  double delay = rxd ? (dsum / rxd) * 1000.0 : 0.0;
  double hops = rx ? (double)fwd / rx + 1.0 : 0.0;
  std::ofstream o(g.outDir + "/summary.csv");
  o << "nodes,areaX,areaY,sim_time,seed,standard,hello,route_timeout,"
       "n_flows,pkt_rate,tx,rx,pdr,delay_ms,mean_hops\n";
  o << g.nNodes << "," << g.areaX << "," << g.areaY << "," << g.simTime << ","
    << g.seed << "," << g.standard << "," << (g.enableHello?1:0) << ","
    << g.routeTimeout << "," << g.nFlows << "," << g.pktRate << ","
    << tx << "," << rx << "," << std::fixed << std::setprecision(6) << pdr
    << "," << delay << "," << hops << "\n";
  o.close();
  std::cout << "  bhSeen=" << aodvatk::GetBhSeen() << " bhDrops=" << aodvatk::GetBhDrops() << std::endl;
  std::cout << "[n=" << g.nNodes << " " << g.standard << " hello="
            << (g.enableHello?1:0) << "] PDR=" << std::fixed
            << std::setprecision(4) << pdr << "  delay=" << std::setprecision(1)
            << delay << "ms  hops=" << std::setprecision(2) << hops
            << "  tx=" << tx << " rx=" << rx << std::endl;
  g_feat.close();
  Simulator::Destroy();
  return 0;
}
