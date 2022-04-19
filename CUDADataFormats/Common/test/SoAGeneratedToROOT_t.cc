/*
 * SoAStreamer_t.cu
 * 
 * A test validating and the serialization of SoA Layouts to a ROOT file
 */

#include <TFile.h>
#include <TTree.h>
#include <memory>
#include "SoAGeneratedToROOT.h"


using SoALayout = SoALayoutTemplate<>;
using SoAView =
    SoAViewTemplate<cms::soa::CacheLineSize::NvidiaGPU, cms::soa::AlignmentEnforcement::Relaxed>;

struct SoABufferAndLayout {
  SoALayout layout_;
  SoAView view_;
  using AlignedBuffer = std::unique_ptr<std::byte, decltype(std::free) *>;
  AlignedBuffer buffer_ = AlignedBuffer(nullptr, std::free);
  
  void Allocate(size_t nElements) {
    buffer_.reset(reinterpret_cast<std::byte*>(aligned_alloc(SoALayout::byteAlignment, SoALayout::computeDataSize(nElements)))); 
    // , ) = std::make_unique<std::byte[]>(SoALayout::computeDataSize(nElements));
    new (&layout_)  SoALayout(buffer_.get(), nElements);
    new (&view_) SoAView(layout_);
  }
  
  void Fill() {
    for (size_t i=0; i<view_.soaMetadata().size(); i++) {
      auto vi = view_[i];
      vi.x() = 42 + i;
      vi.y() = 42 - i;
      vi.z() = 24 + i;
    }
  }
  
  void Check() {
    if (!view_.soaMetadata().size()) { std::cout << "Empty view! Cannot check." << std::endl; abort();  }
    for (size_t i=0; i<view_.soaMetadata().size(); i++) {
      auto vi = view_[i];
      if (vi.x() != 42 + i) { std::cout << "x mismatch at i=" << i << "(" << vi.x() << "/" << 42 + i << ")" << std::endl; abort(); }
      if(vi.y() != 42 - i) { std::cout << "y mismatch at i=" << i << "(" << vi.y() << "/" << 42 - i << ")" << std::endl; abort(); }
      if(vi.z() != 24 + i){ std::cout << "x mismatch at i=" << i << "(" << vi.z() << "/" << 24 + i << ")" << std::endl; abort(); }
    }
  }
  
  void Dump() {
    // We are interested in the layout, which is the element going to disk.
    std::cout << "size=" << view_.soaMetadata().size() //<<  " buffer=" << view_.soaMetadata().data() 
            << " x=" << view_.soaMetadata().addressOf_x() << " y=" << view_.soaMetadata().addressOf_y() << " (y - x)="
    << reinterpret_cast<std::byte*>(
            reinterpret_cast<intptr_t>(view_.soaMetadata().addressOf_x()) 
            - reinterpret_cast<intptr_t>(view_.soaMetadata().addressOf_y()))
    << " buffer size=" << SoALayout::computeDataSize(view_.soaMetadata().size()) 
    << "(0x" << std::hex << SoALayout::computeDataSize(view_.soaMetadata().size())
    << ")" << std::endl;
  }
};

void writeSoA() {
  std::cout << "write begin" << std::endl << std::flush;
  constexpr size_t nElements = 128;

  // Allocate the buffer/view/layout
  SoABufferAndLayout soaBL;
  soaBL.Allocate(nElements);
  soaBL.Dump();
  soaBL.Fill();
  soaBL.Check();
  
  std::unique_ptr<TFile> myFile( TFile::Open("SoAGeneratedToROOT_t.root", "RECREATE") );
  TTree tt("serializerNoTObjTree",  "A SoA TTree");
  // In CMSSW, we will get a branch of objects (each row from the branched corresponding to an event)
  // So we have a branch with one element for the moment.
  [[maybe_unused]] auto Branch = tt.Branch("SoALayout", &soaBL.layout_);
  std::cout << "In writeFile(), about to Fill()" << std::endl;
  soaBL.Dump();
  auto prevGDebug = gDebug;
  //gDebug=5;
  tt.Fill();
  gDebug=prevGDebug;
  tt.Write();
  myFile->Close();
  std::cout << "write end" << std::endl << std::flush;
}

void readSoA() {
  std::cout << "read begin" << std::endl << std::flush;
  std::unique_ptr<TFile> myFile( TFile::Open("SoAGeneratedToROOT_t.root", "READ") );
  myFile->ls();
  std::unique_ptr<TTree> fakeSoATree((TTree*)myFile->Get("serializerNoTObjTree"));
  fakeSoATree->ls();
  auto prevGDebug = gDebug;
  gDebug = 3;
  SoALayout* soal = nullptr;
  fakeSoATree->SetBranchAddress("SoALayout", &soal);
  fakeSoATree->GetEntry(0);
  gDebug = prevGDebug;
  std::cout << "fakeSoAAddress=" << soal << std::endl;
  assert (soal);
  SoABufferAndLayout soabl;
  new (&soabl.view_) SoAView(*soal);
  soabl.Dump();
  //fakeSoA->DumpData();
  std::cout << "Checking SoA readback...";
  soabl.Check();
  std::cout << " OK" << std::endl << std::flush;
}

int main() {
  writeSoA();
  readSoA();
  return EXIT_SUCCESS;
}