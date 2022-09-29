#include <cassert>

#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

class TestAlpakaAnalyzer : public edm::stream::EDAnalyzer<> {
public:
  TestAlpakaAnalyzer(edm::ParameterSet const& config)
      : source_{config.getParameter<edm::InputTag>("source")}, token_{consumes(source_)} {}

  void analyze(edm::Event const& event, edm::EventSetup const&) override {
    portabletest::TestHostCollection const& product = event.get(token_);

    auto const& view = product.const_view();
    const portabletest::M36d m36d{{1,2,3,4,5,6}, {2,4,6,8,10,12}, {3,6,9,12,15,18}};
    assert (view.r() == 1.);
    for (int32_t i = 0; i < view.metadata().size(); ++i) {
      auto vi = view[i];
      assert(vi.x() == 0.);
      assert(vi.y() == 0.);
      assert(vi.z() == 0.);
      assert(vi.id() == i);
      assert(vi.m() == i * m36d);
    }

    edm::LogInfo msg("TestAlpakaAnalyzer");
    msg << source_.encode() << ".size() = " << view.metadata().size() << '\n';
    msg << "  data = " << product.buffer().data() << ",\n"
        << "  x    = " << view.metadata().addressOf_x() << ",\n"
        << "  y    = " << view.metadata().addressOf_y() << ",\n"
        << "  z    = " << view.metadata().addressOf_z() << ",\n"
        << "  id   = " << view.metadata().addressOf_id() << ",\n"
        << "  r    = " << view.metadata().addressOf_r() << ",\n"
        << "  m    = " << view.metadata().addressOf_m() << '\n';
    msg << std::hex << "  [y - x] = 0x"
        << reinterpret_cast<intptr_t>(view.metadata().addressOf_y()) -
               reinterpret_cast<intptr_t>(view.metadata().addressOf_x())
        << "  [z - y] = 0x"
        << reinterpret_cast<intptr_t>(view.metadata().addressOf_z()) -
               reinterpret_cast<intptr_t>(view.metadata().addressOf_y())
        << "  [id - z] = 0x"
        << reinterpret_cast<intptr_t>(view.metadata().addressOf_id()) -
               reinterpret_cast<intptr_t>(view.metadata().addressOf_z())
        << "  [r - id] = 0x"
        << reinterpret_cast<intptr_t>(view.metadata().addressOf_r()) -
               reinterpret_cast<intptr_t>(view.metadata().addressOf_id())
        << "  [m - r] = 0x"
        << reinterpret_cast<intptr_t>(view.metadata().addressOf_m()) -
               reinterpret_cast<intptr_t>(view.metadata().addressOf_r());
    if (event.id().event() == 1) msg << '\n' <<  view[2].m();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::InputTag source_;
  const edm::EDGetTokenT<portabletest::TestHostCollection> token_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestAlpakaAnalyzer);
