#include "EventFilter/CSCRawToDigi/interface/CSCDCCUnpacker.h"

//Framework stuff
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
//FEDRawData
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"


//Digi stuff
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCFEBStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"

#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCFEBStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDCCStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCDCCStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigiCollection.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCLCTData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCRPCData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"
#include <EventFilter/CSCRawToDigi/interface/CSCMonitorInterface.h>
#include "FWCore/ServiceRegistry/interface/Service.h"



#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

CSCDCCUnpacker::CSCDCCUnpacker(const edm::ParameterSet & pset) :
  numOfEvents(0) {
  printEventNumber = pset.getUntrackedParameter<bool>("PrintEventNumber", true);
  debug = pset.getUntrackedParameter<bool>("Debug", false);
  useExaminer = pset.getUntrackedParameter<bool>("UseExaminer", true);
  examinerMask = pset.getUntrackedParameter<unsigned int>("ExaminerMask",0x7FB7BF6);
  instatiateDQM = pset.getUntrackedParameter<bool>("runDQM", false);
  errorMask = pset.getUntrackedParameter<unsigned int>("ErrorMask",0xDFCFEFFF);
  unpackStatusDigis = pset.getUntrackedParameter<bool>("UnpackStatusDigis", false);
  inputObjectsTag = pset.getParameter<edm::InputTag>("InputObjects");
  
  // Visualization of raw data in FED-less events 
  visualFEDInspect=pset.getUntrackedParameter<bool>("VisualFEDInspect", false);
  visualFEDShort=pset.getUntrackedParameter<bool>("VisualFEDShort", false);
  //Visualization of raw data in FED-less events 
  
  // Enable Format Status Digis
  useFormatStatus = pset.getUntrackedParameter<bool>("UseFormatStatus", false);

  // Selective unpacking mode will skip only troublesome CSC blocks and not whole DCC/DDU block
  useSelectiveUnpacking = pset.getUntrackedParameter<bool>("UseSelectiveUnpacking", false);

  if(instatiateDQM)  {
    monitor = edm::Service<CSCMonitorInterface>().operator->();
  }
  
  produces<CSCWireDigiCollection>("MuonCSCWireDigi");
  produces<CSCStripDigiCollection>("MuonCSCStripDigi");
  produces<CSCComparatorDigiCollection>("MuonCSCComparatorDigi");
  produces<CSCALCTDigiCollection>("MuonCSCALCTDigi");
  produces<CSCCLCTDigiCollection>("MuonCSCCLCTDigi");
  produces<CSCRPCDigiCollection>("MuonCSCRPCDigi");
  produces<CSCCorrelatedLCTDigiCollection>("MuonCSCCorrelatedLCTDigi");
  
  if (unpackStatusDigis)  {
    produces<CSCCFEBStatusDigiCollection>("MuonCSCCFEBStatusDigi");
    produces<CSCTMBStatusDigiCollection>("MuonCSCTMBStatusDigi");
    produces<CSCDMBStatusDigiCollection>("MuonCSCDMBStatusDigi");
    produces<CSCALCTStatusDigiCollection>("MuonCSCALCTStatusDigi");
    produces<CSCDDUStatusDigiCollection>("MuonCSCDDUStatusDigi");
    produces<CSCDCCStatusDigiCollection>("MuonCSCDCCStatusDigi");
  }

  if (useFormatStatus) {
    produces<CSCDCCFormatStatusDigiCollection>("MuonCSCDCCFormatStatusDigi");
  }
  //CSCAnodeData::setDebug(debug);
  CSCALCTHeader::setDebug(debug);
  CSCCLCTData::setDebug(debug);
  CSCEventData::setDebug(debug);
  CSCTMBData::setDebug(debug);
  CSCDCCEventData::setDebug(debug);
  CSCDDUEventData::setDebug(debug);
  CSCTMBHeader::setDebug(debug);
  CSCRPCData::setDebug(debug);
  CSCDDUEventData::setErrorMask(errorMask);
  
}

CSCDCCUnpacker::~CSCDCCUnpacker(){ 
  //fill destructor here
}


void CSCDCCUnpacker::produce(edm::Event & e, const edm::EventSetup& c){

  ///access database for mapping
  // Do we really have to do this every event???
  // ... Yes, because framework is more efficient than you are at caching :)
  // (But if you want to actually DO something specific WHEN the mapping changes, check out ESWatcher)
  edm::ESHandle<CSCCrateMap> hcrate;
  c.get<CSCCrateMapRcd>().get(hcrate); 
  const CSCCrateMap* pcrate = hcrate.product();


  if (printEventNumber) ++numOfEvents;

  /// Get a handle to the FED data collection
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByLabel(inputObjectsTag, rawdata);
    
  /// create the collections of CSC digis
  std::auto_ptr<CSCWireDigiCollection> wireProduct(new CSCWireDigiCollection);
  std::auto_ptr<CSCStripDigiCollection> stripProduct(new CSCStripDigiCollection);
  std::auto_ptr<CSCALCTDigiCollection> alctProduct(new CSCALCTDigiCollection);
  std::auto_ptr<CSCCLCTDigiCollection> clctProduct(new CSCCLCTDigiCollection);
  std::auto_ptr<CSCComparatorDigiCollection> comparatorProduct(new CSCComparatorDigiCollection);
  std::auto_ptr<CSCRPCDigiCollection> rpcProduct(new CSCRPCDigiCollection);
  std::auto_ptr<CSCCorrelatedLCTDigiCollection> corrlctProduct(new CSCCorrelatedLCTDigiCollection);
  std::auto_ptr<CSCCFEBStatusDigiCollection> cfebStatusProduct(new CSCCFEBStatusDigiCollection);
  std::auto_ptr<CSCDMBStatusDigiCollection> dmbStatusProduct(new CSCDMBStatusDigiCollection);
  std::auto_ptr<CSCTMBStatusDigiCollection> tmbStatusProduct(new CSCTMBStatusDigiCollection);
  std::auto_ptr<CSCDDUStatusDigiCollection> dduStatusProduct(new CSCDDUStatusDigiCollection);
  std::auto_ptr<CSCDCCStatusDigiCollection> dccStatusProduct(new CSCDCCStatusDigiCollection);
  std::auto_ptr<CSCALCTStatusDigiCollection> alctStatusProduct(new CSCALCTStatusDigiCollection);

  std::auto_ptr<CSCDCCFormatStatusDigiCollection> formatStatusProduct(new CSCDCCFormatStatusDigiCollection);
 

  // If set selective unpacking mode 
  // hardcoded examiner mask below to check for DCC and DDU level errors will be used first
  // then examinerMask for CSC level errors will be used during unpacking of each CSC block
  unsigned long dccBinCheckMask = 0x06080016;

  for (int id=FEDNumbering::MINCSCFEDID;
       id<=FEDNumbering::MAXCSCFEDID; ++id) { // loop over DCCs
    /// uncomment this for regional unpacking
    /// if (id!=SOME_ID) continue;
    
    /// Take a reference to this FED's data
    const FEDRawData& fedData = rawdata->FEDData(id);
    unsigned long length =  fedData.size();
  

    if (length>=32){ ///if fed has data then unpack it
      CSCDCCExaminer* examiner = NULL;
      std::stringstream examiner_out, examiner_err;
      goodEvent = true;
      if (useExaminer) {///examine event for integrity
	// CSCDCCExaminer examiner;
	examiner = new CSCDCCExaminer();
	examiner->output1().redirect(examiner_out);
	examiner->output2().redirect(examiner_err);
	if( examinerMask&0x40000 ) examiner->crcCFEB(1);
	if( examinerMask&0x8000  ) examiner->crcTMB (1);
	if( examinerMask&0x0400  ) examiner->crcALCT(1);
	examiner->output1().show();
	examiner->output2().show();
	examiner->setMask(examinerMask);
	const short unsigned int *data = (short unsigned int *)fedData.data();


	// Event data hex dump
	/*short unsigned * buf = (short unsigned int *)fedData.data();
	  std::cout <<std::endl<<length/2<<" words of data:"<<std::endl;
	  for (int i=0;i<length/2;i++) {
	  printf("%04x %04x %04x %04x\n",buf[i+3],buf[i+2],buf[i+1],buf[i]);
	  i+=3;
	  }*/


        int res = examiner->check(data,long(fedData.size()/2));
	if( res < 0 )	{
	  goodEvent=false;
	} 
	else {	  
	  if (useSelectiveUnpacking)  goodEvent=!(examiner->errors()&dccBinCheckMask);
	  else goodEvent=!(examiner->errors()&examinerMask);
	}

	/*
         std::cout << "FED" << id << " " << fedData.size() << " " << goodEvent << " " 
	<< std::hex << examiner->errors() << std::dec << " " << status << std::endl;
	*/

	// Fill Format status digis per FED
	// Remove examiner->errors() != 0 check if we need to put status digis for every event
	if (useFormatStatus && (examiner->errors() !=0))
	  // formatStatusProduct->insertDigi(CSCDetId(1,1,1,1,1), CSCDCCFormatStatusDigi(id,examiner,dccBinCheckMask));
	  formatStatusProduct->insertDigi(CSCDetId(1,1,1,1,1), 
		CSCDCCFormatStatusDigi(id,dccBinCheckMask,
			examiner->getMask(), 
			examiner->errors(),
			examiner->errorsDetailedDDU(),
			examiner->errorsDetailed(),
			examiner->payloadDetailed(),
			examiner->statusDetailed()));
	}	
     
      //Visualization of raw data in FED-less events 
      if(visualFEDInspect){
	if (!goodEvent){
	  short unsigned * buf = (short unsigned int *)fedData.data();
          visual_raw(length/2, id,(int)e.id().run(),(int)e.id().event(), visualFEDShort, buf);
        }
      }
	          
      if (goodEvent) {
	///get a pointer to data and pass it to constructor for unpacking


	CSCDCCExaminer * ptrExaminer = examiner;
	if (!useSelectiveUnpacking) ptrExaminer = NULL;
		
  	CSCDCCEventData dccData((short unsigned int *) fedData.data(),ptrExaminer);
	
	if(instatiateDQM) monitor->process(examiner, &dccData);
	
	///get a reference to dduData
	const std::vector<CSCDDUEventData> & dduData = dccData.dduData();
	
	/// set default detid to that for E=+z, S=1, R=1, C=1, L=1
	CSCDetId layer(1, 1, 1, 1, 1);
	
	if (unpackStatusDigis) 
	  dccStatusProduct->insertDigi(layer, CSCDCCStatusDigi(dccData.dccHeader().data(),
							       dccData.dccTrailer().data(),
							       examiner->errors()));

	for (unsigned int iDDU=0; iDDU<dduData.size(); ++iDDU) {  // loop over DDUs
	  /// skip the DDU if its data has serious errors
	  /// define a mask for serious errors 
	  if (dduData[iDDU].trailer().errorstat()&errorMask) {
	    LogTrace("CSCDCCUnpacker|CSCRawToDigi") << "DDU# " << iDDU << " has serious error - no digis unpacked! " <<
	      std::hex << dduData[iDDU].trailer().errorstat();
	    continue; // to next iteration of DDU loop
	  }
		  
	  if (unpackStatusDigis) dduStatusProduct->
				   insertDigi(layer, CSCDDUStatusDigi(dduData[iDDU].header().data(),
								      dduData[iDDU].trailer().data()));
	  
	  ///get a reference to chamber data
	  const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();

	  for (unsigned int iCSC=0; iCSC<cscData.size(); ++iCSC) { // loop over CSCs

	    ///first process chamber-wide digis such as LCT
	    int vmecrate = cscData[iCSC].dmbHeader()->crateID();
	    int dmb = cscData[iCSC].dmbHeader()->dmbID();

	    int icfeb = 0;  /// default value for all digis not related to cfebs
	    int ilayer = 0; /// layer=0 flags entire chamber

	    if (debug)
              LogTrace ("CSCDCCUnpacker|CSCRawToDigi") << "crate = " << vmecrate << "; dmb = " << dmb;
	    
	    if ((vmecrate>=1)&&(vmecrate<=60) && (dmb>=1)&&(dmb<=10)&&(dmb!=6)) {
	      layer = pcrate->detId(vmecrate, dmb,icfeb,ilayer );
	    } 
	    else{
	      LogTrace ("CSCDCCUnpacker|CSCRawToDigi") << " detID input out of range!!! ";
              LogTrace ("CSCDCCUnpacker|CSCRawToDigi")
                << " skipping chamber vme= " << vmecrate << " dmb= " << dmb;
	      continue; // to next iteration of iCSC loop
	    }


	    /// check alct data integrity 
	    int nalct = cscData[iCSC].dmbHeader()->nalct();
	    bool goodALCT=false;
	    //if (nalct&&(cscData[iCSC].dataPresent>>6&0x1)==1) {
	    if (nalct&&cscData[iCSC].alctHeader()) {  
	      if (cscData[iCSC].alctHeader()->check()){
		goodALCT=true;
	      }
	      else {
		LogTrace ("CSCDCCUnpacker|CSCRawToDigi") <<
		  "not storing ALCT digis; alct is bad or not present";
	      }
	    } else {
	      if (debug) LogTrace ("CSCDCCUnpacker|CSCRawToDigi") << "nALCT==0 !!!";
	    }

	    /// fill alct digi
	    if (goodALCT){
	      std::vector <CSCALCTDigi>  alctDigis =
		cscData[iCSC].alctHeader()->ALCTDigis();
	      alctProduct->put(std::make_pair(alctDigis.begin(), alctDigis.end()),layer);
	    }
		    
		  
	    ///check tmb data integrity
	    int nclct = cscData[iCSC].dmbHeader()->nclct();
	    bool goodTMB=false;
	    //	    if (nclct&&(cscData[iCSC].dataPresent>>5&0x1)==1) {
	    if (nclct&&cscData[iCSC].tmbData()) {
	      if (cscData[iCSC].tmbHeader()->check()){
		if (cscData[iCSC].clctData()->check()) goodTMB=true; 
	      }
	      else {
		LogTrace ("CSCDCCUnpacker|CSCRawToDigi") <<
		  "one of TMB checks failed! not storing TMB digis ";
	      }
	    }
	    else {
	      if (debug) LogTrace ("CSCDCCUnpacker|CSCRawToDigi") << "nCLCT==0 !!!";
	    }
		      
	    /// fill correlatedlct and clct digis
	    if (goodTMB) { 
	      std::vector <CSCCorrelatedLCTDigi>  correlatedlctDigis =
		cscData[iCSC].tmbHeader()->CorrelatedLCTDigis(layer.rawId());
	      
	      corrlctProduct->put(std::make_pair(correlatedlctDigis.begin(),
						 correlatedlctDigis.end()),layer);
		      
	      std::vector <CSCCLCTDigi>  clctDigis =
		cscData[iCSC].tmbHeader()->CLCTDigis(layer.rawId());
	      clctProduct->put(std::make_pair(clctDigis.begin(), clctDigis.end()),layer);
	      
	      /// fill cscrpc digi
	      if (cscData[iCSC].tmbData()->checkSize()) {
		if (cscData[iCSC].tmbData()->hasRPC()) {
		  std::vector <CSCRPCDigi>  rpcDigis =
		    cscData[iCSC].tmbData()->rpcData()->digis();
		  rpcProduct->put(std::make_pair(rpcDigis.begin(), rpcDigis.end()),layer);
		}
	      } 
	      else LogTrace("CSCDCCUnpacker|CSCRawToDigi") <<" TMBData check size failed!";
	    }
		    
		  
	    /// fill cfeb status digi
	    if (unpackStatusDigis) {
	      for ( icfeb = 0; icfeb < 5; ++icfeb ) {///loop over status digis
		if ( cscData[iCSC].cfebData(icfeb) != NULL )
		  cfebStatusProduct->
		    insertDigi(layer, cscData[iCSC].cfebData(icfeb)->statusDigi());
	      }
	      /// fill dmb status digi
	      dmbStatusProduct->insertDigi(layer, CSCDMBStatusDigi(cscData[iCSC].dmbHeader()->data(),
								   cscData[iCSC].dmbTrailer()->data()));
	      if (goodTMB)  tmbStatusProduct->
			      insertDigi(layer, CSCTMBStatusDigi(cscData[iCSC].tmbHeader()->data(),
								 cscData[iCSC].tmbData()->tmbTrailer()->data()));
	      if (goodALCT) alctStatusProduct->
			      insertDigi(layer, CSCALCTStatusDigi(cscData[iCSC].alctHeader()->data(),
								  cscData[iCSC].alctTrailer()->data()));
	    }
		

	    /// fill wire, strip and comparator digis...
	    for (int ilayer = 1; ilayer <= 6; ++ilayer) {
	      /// set layer, dmb and vme are valid because already checked in line 240
	      // (You have to be kidding. Line 240 in whose universe?)

	      // Allocate all ME1/1 wire digis to ring 1
	      layer = pcrate->detId( vmecrate, dmb, 0, ilayer );

	      std::vector <CSCWireDigi> wireDigis =  cscData[iCSC].wireDigis(ilayer);
	      
	      wireProduct->put(std::make_pair(wireDigis.begin(), wireDigis.end()),layer);
	      
	      for ( icfeb = 0; icfeb < 5; ++icfeb ) {
		layer = pcrate->detId( vmecrate, dmb, icfeb,ilayer );
		if (cscData[iCSC].cfebData(icfeb)) {
		  std::vector<CSCStripDigi> stripDigis;
		  cscData[iCSC].cfebData(icfeb)->digis(layer.rawId(),stripDigis);
		  stripProduct->put(std::make_pair(stripDigis.begin(), 
						   stripDigis.end()),layer);
		}
			      
		if (goodTMB){
		  std::vector <CSCComparatorDigi>  comparatorDigis =
		    cscData[iCSC].clctData()->comparatorDigis(layer.rawId(), icfeb);
		  // Set cfeb=0, so that ME1/a and ME1/b comparators go to
		  // ring 1.
		  layer = pcrate->detId( vmecrate, dmb, 0, ilayer );
		  comparatorProduct->put(std::make_pair(comparatorDigis.begin(), 
							comparatorDigis.end()),layer);
		}
	      } // end of loop over cfebs
	    } // end of loop over layers
	  } // end of loop over chambers
	} // endof loop over DDUs
      } // end of good event
      else   {
        LogTrace("CSCDCCUnpacker|CSCRawToDigi") <<
          "ERROR! Examiner rejected FED #" << id;
        if (examiner) {
          for (int i=0; i<examiner->nERRORS; ++i) {
            if (((examinerMask&examiner->errors())>>i)&0x1)
              LogTrace("CSCDCCUnpacker|CSCRawToDigi")<<examiner->errName(i);
          }
          if (debug) {
            LogTrace("CSCDCCUnpacker|CSCRawToDigi")
              << " Examiner errors:0x" << std::hex << examiner->errors()
              << " & 0x" << examinerMask
              << " = " << (examiner->errors()&examinerMask);
            if (examinerMask&examiner->errors()) {
              LogTrace("CSCDCCUnpacker|CSCRawToDigi")
                << "Examiner output: " << examiner_out.str();
              LogTrace("CSCDCCUnpacker|CSCRawToDigi")
                << "Examiner errors: " << examiner_err.str();
            }
          }
        }

	// dccStatusProduct->insertDigi(CSCDetId(1,1,1,1,1), CSCDCCStatusDigi(examiner->errors()));
	// if(instatiateDQM)  monitor->process(examiner, NULL);
      }
      if (examiner!=NULL) delete examiner;
    } // end of if fed has data
  } // end of loop over DCCs
  // put into the event
  e.put(wireProduct,          "MuonCSCWireDigi");
  e.put(stripProduct,         "MuonCSCStripDigi");
  e.put(alctProduct,          "MuonCSCALCTDigi");
  e.put(clctProduct,          "MuonCSCCLCTDigi");
  e.put(comparatorProduct,    "MuonCSCComparatorDigi");
  e.put(rpcProduct,           "MuonCSCRPCDigi");
  e.put(corrlctProduct,       "MuonCSCCorrelatedLCTDigi");

  if (useFormatStatus)  e.put(formatStatusProduct,    "MuonCSCDCCFormatStatusDigi");

  if (unpackStatusDigis)
    {
      e.put(cfebStatusProduct,    "MuonCSCCFEBStatusDigi");
      e.put(dmbStatusProduct,     "MuonCSCDMBStatusDigi");
      e.put(tmbStatusProduct,     "MuonCSCTMBStatusDigi");
      e.put(dduStatusProduct,     "MuonCSCDDUStatusDigi");
      e.put(dccStatusProduct,     "MuonCSCDCCStatusDigi");
      e.put(alctStatusProduct,    "MuonCSCALCTStatusDigi");
    }
  if (printEventNumber) LogTrace("CSCDCCUnpacker|CSCRawToDigi") 
   <<"[CSCDCCUnpacker]: " << numOfEvents << " events processed ";
}


// *** Visualization of raw data in FED-less events (begin) ***

void CSCDCCUnpacker::visual_raw(int hl,int id, int run, int event,bool fedshort,short unsigned int *buf) const {

	LogTrace("badData") << std::endl << std::endl;
	LogTrace("badData") << "Run: "<< run << " Event: " << event;
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "Problem seems in FED-" << id << "  " << "(scroll down to see summary)";
	LogTrace("badData") <<"********************************************************************************";
	LogTrace("badData") <<hl<<" words of data:";
	
	//================================================
	// FED codes in DCC
	std::vector<int> dcc_id;
	int dcc_h1_id=0;
	for (int i=750;i<758;i++)
	    dcc_id.push_back(i);
	char dcc_common[]="DCC-";    

	//================================================
	// DDU codes per FED
	std::vector<int> ddu_id;
	int ddu_h1_12_13=0;
	for (int i=1;i<37;i++)
	    ddu_id.push_back(i);
	// For DDU Headers and tarailers
	char ddu_common[]="DDU-";
	char ddu_header1[]="Header 1";
	char ddu_header2[]="Header 2";
	char ddu_header3[]="Header 3";
	char ddu_trail1[]="Trailer 1", ddu_trail2[]="Trailer 2", ddu_trail3[]="Trailer 3";
	// For Header 2
	char ddu_header2_bit[]={'8','0','0','0','0','0','0','1','8','0','0','0','8','0','0','0'};
	char ddu_trailer1_bit[]={'8','0','0','0','f','f','f','f','8','0','0','0','8','0','0','0'};
	char ddu_trailer3_bit[]={'a'};
	// Corrupted Trailers
	char ddu_tr1_err_common[]="Incomplet";
	//====================================================
	
	//DMB
	char dmb_common[]="DMB", dmb_header1[]="Header 1", dmb_header2[]="Header 2";
	char dmb_common_crate[]="crate:", dmb_common_slot[]="slot:"; 
	char dmb_common_l1a[]="L1A:";
	char dmb_header1_bit[]={'9','9','9','9'};
	char dmb_header2_bit[]={'a','a','a','a'};
	char dmb_tr1[]="Trailer 1", dmb_tr2[]="Trailer 2";
	char dmb_tr1_bit[]={'f','f','f','f'}, dmb_tr2_bit[]={'e','e','e','e'};
	
	
	//=====================================================
	
	// ALCT
	char alct_common[]="ALCT", alct_header1[]="Header 1", alct_header2[]="Header 2";
	char alct_common_bxn[]="BXN:";
	char alct_common_wcnt2[]="| Actual word count:"; 
	char alct_common_wcnt1[]="Expected word count:";
	char alct_header1_bit[]={'d','d','d','d','b','0','a'};
	char alct_header2_bit[]={'0','0','0','0'};
	char alct_tr1[]="Trailer 1"; 
	
	//======================================================

	//TMB
	char tmb_common[]="TMB", tmb_header1[]="Header", tmb_tr1[]="Trailer";
	char tmb_header1_bit[]={'d','d','d','d','b','0','c'};
	char tmb_tr1_bit[]={'d','d','d','d','e','0','f'};
	
	//======================================================
	
	//CFEB 
	char cfeb_common[]="CFEB", cfeb_tr1[]="Trailer", cfeb_b[]="B-word";
	char cfeb_common_sample[]="sample:";
	
	//======================================================	
	
	//Auxiliary variables 
	
	// Bufers
	int word_lines=hl/4;
	char tempbuf[80];
	char tempbuf1[80];
	char tempbuf_short[17];
	char sign1[]="  --->| ";
	
	// Counters
	int word_numbering=0;
	int ddu_inst_i=0, ddu_inst_n=0, ddu_inst_l1a=0;
	int dmb_inst_crate=0, dmb_inst_slot=0, dmb_inst_l1a=0;
	int cfeb_sample=0;
	int alct_inst_l1a=0;
	int alct_inst_bxn=0;
	int alct_inst_wcnt1=0;
	int alct_inst_wcnt2=0;
	int alct_start=0;
	int alct_stop=0;
	int tmb_inst_l1a=0;
	int tmb_inst_wcnt1=0;
	int tmb_inst_wcnt2=0;
	int tmb_start=0;
	int tmb_stop=0;
	int dcc_h1_check=0;
	
	//Flags
	int ddu_h2_found=0;  //DDU Header 2 found
	int w=0;
	
	//Logic variables
	const int sz1=5;
	bool ddu_h2_check[sz1];
	bool ddu_h1_check=false;
	bool dmb_h1_check[sz1];
	bool dmb_h2_check[sz1];
	bool ddu_h2_h1=false;
	bool ddu_tr1_check[sz1];
	bool alct_h1_check[sz1];
	bool alct_h2_check[sz1];
	bool alct_tr1_check[sz1];
	bool dmb_tr1_check[sz1];
	bool dmb_tr2_check[sz1];
	bool tmb_h1_check[sz1];
	bool tmb_tr1_check[sz1];
	bool cfeb_tr1_check[sz1];
	bool cfeb_b_check[sz1];
	bool ddu_tr1_bad_check[sz1];
	bool extraction=fedshort;
	
	//Summary vectors
	//DDU
	std::vector<int> ddu_h1_coll;
	std::vector<int> ddu_h1_n_coll;
	std::vector<int> ddu_h2_coll;
	std::vector<int> ddu_h3_coll;
	std::vector<int> ddu_t1_coll;
	std::vector<int> ddu_t2_coll;
	std::vector<int> ddu_t3_coll;
	std::vector<int> ddu_l1a_coll;
	//DMB
	std::vector<int> dmb_h1_coll;
	std::vector<int> dmb_h2_coll;
	std::vector<int> dmb_t1_coll;
	std::vector<int> dmb_t2_coll;
	std::vector<int> dmb_crate_coll;
	std::vector<int> dmb_slot_coll;
	std::vector<int> dmb_l1a_coll;
	//ALCT
	std::vector<int> alct_h1_coll;
	std::vector<int> alct_h2_coll;
	std::vector<int> alct_t1_coll;
	std::vector<int> alct_l1a_coll;
	std::vector<int> alct_bxn_coll;
	std::vector<int> alct_wcnt1_coll;
	std::vector<int> alct_wcnt2_coll;
	std::vector<int> alct_wcnt2_id_coll;
	//TMB
	std::vector<int> tmb_h1_coll;
	std::vector<int> tmb_t1_coll;
	std::vector<int> tmb_l1a_coll;
	std::vector<int> tmb_wcnt1_coll;
	std::vector<int> tmb_wcnt2_coll;
	//CFEB
	std::vector<int> cfeb_t1_coll;
	
	//========================================================
	
	// DCC Header and Ttrailer information
	char dcc_header1[]="DCC Header 1";
	char dcc_header2[]="DCC Header 2";
	char dcc_trail1[]="DCC Trailer 1", dcc_trail1_bit[]={'e'};
	char dcc_trail2[]="DCC Trailer 2", dcc_trail2_bit[]={'a'};
	//=========================================================
        
	for (int i=0;i<hl;i++) {
	// Auxiliary actions
	++word_numbering;
	  for(int j=-1; j<4; j++){
	     if((word_numbering>2)&&(word_numbering<hl/4)){  
	       sprintf(tempbuf_short,"%04x%04x%04x%04x",buf[i+4*(j-1)+3],buf[i+4*(j-1)+2],buf[i+4*(j-1)+1],buf[i+4*(j-1)]);
	        
		ddu_h2_check[j]=((tempbuf_short[0]==ddu_header2_bit[0])&&(tempbuf_short[1]==ddu_header2_bit[1])&&
		                (tempbuf_short[2]==ddu_header2_bit[2])&&(tempbuf_short[3]==ddu_header2_bit[3])&&
		                (tempbuf_short[4]==ddu_header2_bit[4])&&(tempbuf_short[15]==ddu_header2_bit[15])&&
		                (tempbuf_short[5]==ddu_header2_bit[5])&&(tempbuf_short[6]==ddu_header2_bit[6])&&
		                (tempbuf_short[7]==ddu_header2_bit[7])&&(tempbuf_short[8]==ddu_header2_bit[8])//&&
		   //(tempbuf_short[9]==ddu_header2_bit[9])&&(tempbuf_short[10]==ddu_header2_bit[10])&&
		   //(tempbuf_short[11]==ddu_header2_bit[11])&&(tempbuf_short[12]==ddu_header2_bit[12])&&
		   //(tempbuf_short[13]==ddu_header2_bit[13])&&(tempbuf_short[14]==ddu_header2_bit[14])
		 );
		 
		 ddu_tr1_check[j]=((tempbuf_short[0]==ddu_trailer1_bit[0])&&(tempbuf_short[1]==ddu_trailer1_bit[1])&&
	                          (tempbuf_short[2]==ddu_trailer1_bit[2])&&(tempbuf_short[3]==ddu_trailer1_bit[3])&&
		                  (tempbuf_short[4]==ddu_trailer1_bit[4])&&(tempbuf_short[5]==ddu_trailer1_bit[5])&&
		                  (tempbuf_short[6]==ddu_trailer1_bit[6])&&(tempbuf_short[7]==ddu_trailer1_bit[7])&&
				  (tempbuf_short[8]==ddu_trailer1_bit[8])&&(tempbuf_short[9]==ddu_trailer1_bit[9])&&
		                  (tempbuf_short[10]==ddu_trailer1_bit[10])&&(tempbuf_short[11]==ddu_trailer1_bit[11])&&
				  (tempbuf_short[12]==ddu_trailer1_bit[12])&&(tempbuf_short[13]==ddu_trailer1_bit[13])&&
				  (tempbuf_short[14]==ddu_trailer1_bit[14])&&(tempbuf_short[15]==ddu_trailer1_bit[15]));
		 
		 dmb_h1_check[j]=((tempbuf_short[0]==dmb_header1_bit[0])&&(tempbuf_short[4]==dmb_header1_bit[1])&&
		                 (tempbuf_short[8]==dmb_header1_bit[2])&&(tempbuf_short[12]==dmb_header1_bit[3]));
		
		 dmb_h2_check[j]=((tempbuf_short[0]==dmb_header2_bit[0])&&(tempbuf_short[4]==dmb_header2_bit[1])&&
		                 (tempbuf_short[8]==dmb_header2_bit[2])&&(tempbuf_short[12]==dmb_header2_bit[3]));
		 alct_h1_check[j]=((tempbuf_short[0]==alct_header1_bit[0])&&(tempbuf_short[4]==alct_header1_bit[1])&&
		                 (tempbuf_short[8]==alct_header1_bit[2])&&(tempbuf_short[12]==alct_header1_bit[3])&&
				 (tempbuf_short[13]==alct_header1_bit[4])&&(tempbuf_short[14]==alct_header1_bit[5])&&
				 (tempbuf_short[15]==alct_header1_bit[6]));		 
		 alct_h2_check[j]=(((tempbuf_short[0]==alct_header2_bit[0])&&(tempbuf_short[1]==alct_header2_bit[1])&&
		                 (tempbuf_short[2]==alct_header2_bit[2])&&(tempbuf_short[3]==alct_header2_bit[3]))||
				 ((tempbuf_short[4]==alct_header2_bit[0])&&(tempbuf_short[5]==alct_header2_bit[1])&&
		                 (tempbuf_short[6]==alct_header2_bit[2])&&(tempbuf_short[7]==alct_header2_bit[3]))||
				 ((tempbuf_short[8]==alct_header2_bit[0])&&(tempbuf_short[9]==alct_header2_bit[1])&&
		                 (tempbuf_short[10]==alct_header2_bit[2])&&(tempbuf_short[11]==alct_header2_bit[3]))||
				 ((tempbuf_short[12]==alct_header2_bit[0])&&(tempbuf_short[13]==alct_header2_bit[1])&&
		                 (tempbuf_short[14]==alct_header2_bit[2])&&(tempbuf_short[15]==alct_header2_bit[3]))
				 //(tempbuf_short[4]==alct_header2_bit[4])&&(tempbuf_short[5]==alct_header2_bit[5])
				 );
				 // ALCT Trailers
		alct_tr1_check[j]=(((buf[i+4*(j-1)]&0xFFFF)==0xDE0D)&&((buf[i+4*(j-1)+1]&0xF800)==0xD000)&&
		                 ((buf[i+4*(j-1)+2]&0xF800)==0xD000)&&((buf[i+4*(j-1)+3]&0xF000)==0xD000));
				 // DMB Trailers
		 dmb_tr1_check[j]=((tempbuf_short[0]==dmb_tr1_bit[0])&&(tempbuf_short[4]==dmb_tr1_bit[1])&&
		                 (tempbuf_short[8]==dmb_tr1_bit[2])&&(tempbuf_short[12]==dmb_tr1_bit[3]));
		 dmb_tr2_check[j]=((tempbuf_short[0]==dmb_tr2_bit[0])&&(tempbuf_short[4]==dmb_tr2_bit[1])&&
		                 (tempbuf_short[8]==dmb_tr2_bit[2])&&(tempbuf_short[12]==dmb_tr2_bit[3]));
				 // TMB
		 tmb_h1_check[j]=((tempbuf_short[0]==tmb_header1_bit[0])&&(tempbuf_short[4]==tmb_header1_bit[1])&&
		                 (tempbuf_short[8]==tmb_header1_bit[2])&&(tempbuf_short[12]==tmb_header1_bit[3])&&
				 (tempbuf_short[13]==tmb_header1_bit[4])&&(tempbuf_short[14]==tmb_header1_bit[5])&&
				 (tempbuf_short[15]==tmb_header1_bit[6]));
		 tmb_tr1_check[j]=((tempbuf_short[0]==tmb_tr1_bit[0])&&(tempbuf_short[4]==tmb_tr1_bit[1])&&
		                 (tempbuf_short[8]==tmb_tr1_bit[2])&&(tempbuf_short[12]==tmb_tr1_bit[3])&&
				 (tempbuf_short[13]==tmb_tr1_bit[4])&&(tempbuf_short[14]==tmb_tr1_bit[5])&&
				 (tempbuf_short[15]==tmb_tr1_bit[6]));
		                 // CFEB
		 cfeb_tr1_check[j]=(((buf[i+4*(j-1)+1]&0xF000)==0x7000) &&  
		     ((buf[i+4*(j-1)+2]&0xF000)==0x7000) && 
		     (( buf[i+4*(j-1)+1]!= 0x7FFF) || (buf[i+4*(j-1)+2] != 0x7FFF)) &&
		     ((buf[i+4*(j-1)+3] == 0x7FFF) ||   
		     ((buf[i+4*(j-1)+3]&buf[i+4*(j-1)]) == 0x0&&(buf[i+4*(j-1)+3] + buf[i+4*(j-1)] == 0x7FFF ))) );
		 cfeb_b_check[j]=(((buf[i+4*(j-1)+3]&0xF000)==0xB000)&&((buf[i+4*(j-1)+2]&0xF000)==0xB000) && 
		                 ((buf[i+4*(j-1)+1]&0xF000)==0xB000)&&((buf[i+4*(j-1)]=3&0xF000)==0xB000) );   
		                 // DDU Trailers with errors
		 ddu_tr1_bad_check[j]=((tempbuf_short[0]!=ddu_trailer1_bit[0])&&
	                         //(tempbuf_short[1]!=ddu_trailer1_bit[1])&&(tempbuf_short[2]!=ddu_trailer1_bit[2])&&
				 //(tempbuf_short[3]==ddu_trailer1_bit[3])&&
		                 (tempbuf_short[4]!=ddu_trailer1_bit[4])&&
				 //(tempbuf_short[5]==ddu_trailer1_bit[5])&&
		                 //(tempbuf_short[6]==ddu_trailer1_bit[6])&&(tempbuf_short[7]==ddu_trailer1_bit[7])&&
				 (tempbuf_short[8]==ddu_trailer1_bit[8])&&(tempbuf_short[9]==ddu_trailer1_bit[9])&&
		                 (tempbuf_short[10]==ddu_trailer1_bit[10])&&(tempbuf_short[11]==ddu_trailer1_bit[11])&&
				 (tempbuf_short[12]==ddu_trailer1_bit[12])&&(tempbuf_short[13]==ddu_trailer1_bit[13])&&
		                 (tempbuf_short[14]==ddu_trailer1_bit[14])&&(tempbuf_short[15]==ddu_trailer1_bit[15]));	
	  }
	  }
	  // DDU header2 next to header1
	  ddu_h2_h1=ddu_h2_check[2];
	  
	  sprintf(tempbuf_short,"%04x%04x%04x%04x",buf[i+3],buf[i+2],buf[i+1],buf[i]);
	  // Looking for DDU headers 1 per FED   char ddu_16[]="DDU-16", ddu_16_bits_13_14[]={'1','0'};
	  ddu_h1_12_13=(buf[i]>>8);
	  for (int kk=0; kk<36; kk++){	      
	      if(((buf[i+3]&0xF000)==0x5000)&&(ddu_h1_12_13==ddu_id[kk])&&ddu_h2_h1){
	        ddu_h1_coll.push_back(word_numbering); ddu_h1_n_coll.push_back(ddu_id[kk]);
		ddu_inst_l1a=((buf[i+2]&0xFFFF)+((buf[i+3]&0x00FF)<<16));
		ddu_l1a_coll.push_back(ddu_inst_l1a);
	        sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s%i %s%s %s %i",
		word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],
	        sign1,ddu_common,ddu_id[kk],ddu_header1,sign1,dmb_common_l1a,ddu_inst_l1a);
		LogTrace("badData") << tempbuf1; w=0; ddu_h1_check=true; ddu_inst_l1a=0; 
		cfeb_sample=0;
	      }
	  }
	 // Looking for DCC headers and trailers
	 
	 if(((buf[i+3]&0xF000)==0x5000)&&((buf[i]&0x00FF)==0x005F)) {
	     dcc_h1_id=(((buf[i+1]<<12)&0xF000)>>4)+(buf[i]>>8);
	     for(int dcci=0;dcci<8;dcci++){
	         if(dcc_id[dcci]==dcc_h1_id){
	         sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s%i %s",word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],
	         sign1,dcc_common,dcc_h1_id,dcc_header1); dcc_h1_check=word_numbering; break;}
		 else
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x",word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i]);
		 } 
		 LogTrace("badData") << tempbuf1; w=0;
	     }      
	  else if(((word_numbering-1)==dcc_h1_check)&&((buf[i+3]&0xFF00)==0xD900)) {
	     sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s",word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],
	     sign1,dcc_header2);
	     LogTrace("badData") << tempbuf1; w=0; 
	     }
	  else if((word_numbering==word_lines-1)&&(tempbuf_short[0]==dcc_trail1_bit[0])){
	     sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s",word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],
	     sign1,dcc_trail1);
	     LogTrace("badData") << tempbuf1; w=0;
	       }
	  else if((word_numbering==word_lines)&&(tempbuf_short[0]==dcc_trail2_bit[0])){
	     sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s",word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],
	     sign1,dcc_trail2);
	     LogTrace("badData") << tempbuf1; w=0;
	       }
	     // DDU Header 2
	  else if(ddu_h2_check[1]){
               ddu_inst_i = ddu_h1_n_coll.size(); ddu_inst_n=ddu_h1_n_coll[ddu_inst_i-1];
	       sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s%i %s",
	       word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],sign1,ddu_common,
	       ddu_inst_n, ddu_header2);
	       ddu_h2_coll.push_back(word_numbering);
	       LogTrace("badData") << tempbuf1; w=0;
	       ddu_h2_found=1;
	       }
	     // DDU Header 3 (eather between DDU Header 2 DMB Header or DDU Header 2 DDU Trailer1)  
	  else if((ddu_h2_check[0]&&dmb_h1_check[2])||(ddu_h2_check[0]&&ddu_tr1_check[2])){
	     ddu_inst_i = ddu_h1_n_coll.size(); ddu_inst_n=ddu_h1_n_coll[ddu_inst_i-1];
	     sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s%i %s",word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],
	     sign1,ddu_common,ddu_inst_n,ddu_header3);
	     ddu_h3_coll.push_back(word_numbering);
	     LogTrace("badData") << tempbuf1; w=0;
	     ddu_h2_found=0;
	     }
	     // DMB Header 1,2
	     
	    else if(dmb_h1_check[1]){
	         dmb_inst_crate=0; dmb_inst_slot=0; dmb_inst_l1a=0;
		 dmb_inst_l1a=((buf[i]&0x0FFF)+((buf[i+1]&0xFFF)<<12));
		 dmb_l1a_coll.push_back(dmb_inst_l1a);
	       if(dmb_h2_check[2]){
	         dmb_inst_crate=((buf[i+4+1]>>4)&0xFF); dmb_inst_slot=(buf[i+4+1]&0xF);
	         dmb_crate_coll.push_back(dmb_inst_crate); dmb_slot_coll.push_back(dmb_inst_slot);
	       }
	       sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s %s%s%s %i %s %i %s %i",
	       word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],
	       sign1,dmb_common,dmb_header1,sign1,dmb_common_crate,dmb_inst_crate,
	       dmb_common_slot,dmb_inst_slot,dmb_common_l1a,dmb_inst_l1a);
	       dmb_h1_coll.push_back(word_numbering);
	       LogTrace("badData") << tempbuf1; w=0;
	       ddu_h2_found=1;
	       }
	       
	    else if(dmb_h2_check[1]){		 
	       dmb_inst_crate=((buf[i+1]>>4)&0xFF); dmb_inst_slot=(buf[i+1]&0xF);
	       dmb_h2_coll.push_back(word_numbering);
	       if(dmb_h1_check[0])
	          dmb_inst_l1a=((buf[i-4]&0x0FFF)+((buf[i-4+1]&0xFFF)<<12));
	       sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s %s%s%s %i %s %i %s %i",
	       word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],
	       sign1,dmb_common,dmb_header2,sign1,dmb_common_crate,dmb_inst_crate,
	       dmb_common_slot,dmb_inst_slot,dmb_common_l1a,dmb_inst_l1a);
	       LogTrace("badData") << tempbuf1; w=0;
	       ddu_h2_found=1;
	       }
	       
	     //DDU Trailer 1
	   else if(ddu_tr1_check[1]){
	         ddu_inst_i = ddu_h1_n_coll.size(); ddu_inst_n=ddu_h1_n_coll[ddu_inst_i-1];
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s%i %s",
	         word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],sign1,ddu_common,ddu_inst_n,ddu_trail1);
		 ddu_t1_coll.push_back(word_numbering);
	         LogTrace("badData") << tempbuf1; w=0;
		 }

		 //ALCT Header 1,2		
          else if(alct_h1_check[1]){
	         alct_start=word_numbering;
	         alct_inst_l1a=(buf[i+2]&0x0FFF);
		 alct_l1a_coll.push_back(alct_inst_l1a);
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s %s%s %s %i",
	         word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],
		 sign1,alct_common,alct_header1,sign1,dmb_common_l1a,alct_inst_l1a);
		 alct_h1_coll.push_back(word_numbering);
	         LogTrace("badData") << tempbuf1; w=0; alct_inst_l1a=0;
		 }
		 
	  else if((alct_h1_check[0])&&(alct_h2_check[2])) {
	         alct_inst_bxn=(buf[i]&0x0FFF);
		 alct_bxn_coll.push_back(alct_inst_bxn);
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s %s%s%s %i",
	         word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],
		 sign1,alct_common,alct_header2,sign1,alct_common_bxn,alct_inst_bxn);
		 alct_h2_coll.push_back(word_numbering);
	         LogTrace("badData") << tempbuf1; w=0; alct_inst_bxn=0;
		 } 
	         
		 //ALCT Trailer 1
	  else if(alct_tr1_check[1]){
	         alct_stop=word_numbering;
	         if((alct_start!=0)&&(alct_stop!=0)&&(alct_stop>alct_start)) {
		        alct_inst_wcnt2=4*(alct_stop-alct_start+1);
		        alct_wcnt2_coll.push_back(alct_inst_wcnt2);
			alct_wcnt2_id_coll.push_back(alct_start);
			}
	         alct_inst_wcnt1=(buf[i+3]&0x7FF);
		 alct_wcnt1_coll.push_back(alct_inst_wcnt1);
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s %s%s%s %i %s %i",
	         word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],
		 sign1,alct_common,alct_tr1,sign1,alct_common_wcnt1,alct_inst_wcnt1,
		 alct_common_wcnt2,alct_inst_wcnt2);
		 alct_t1_coll.push_back(word_numbering);
	         LogTrace("badData") << tempbuf1; w=0; alct_inst_wcnt1=0;
		 alct_inst_wcnt2=0;  
		 }
		 
		 //DDU Trailer 3
	  else if((ddu_tr1_check[-1])&&(tempbuf_short[0]==ddu_trailer3_bit[0])){
	  //&&(tempbuf_short[0]==ddu_trailer3_bit[0])){
                 ddu_inst_i = ddu_h1_n_coll.size(); ddu_inst_n=ddu_h1_n_coll[ddu_inst_i-1];         
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s%i %s",
	         word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],sign1,ddu_common,ddu_inst_n,ddu_trail3);
		 ddu_t3_coll.push_back(word_numbering);
	         LogTrace("badData") << tempbuf1; w=0;
		 }
		 //DDU Trailer 2
	  else if((ddu_tr1_check[0])&&(tempbuf_short[0]!=ddu_trailer3_bit[0])){
	  //&&(tempbuf_short[0]==ddu_trailer3_bit[0])){
	         ddu_inst_i = ddu_h1_n_coll.size(); ddu_inst_n=ddu_h1_n_coll[ddu_inst_i-1];
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s%i %s",
	         word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],sign1,ddu_common,ddu_inst_n,ddu_trail2);
		 ddu_t2_coll.push_back(word_numbering);
	         LogTrace("badData") << tempbuf1; w=0;
		 } 
	         //DMB Trailer 1,2
         else if(dmb_tr1_check[1]){
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s %s",
	         word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],sign1,dmb_common,dmb_tr1);
		 dmb_t1_coll.push_back(word_numbering);
	         LogTrace("badData") << tempbuf1; w=0;
		 cfeb_sample=0;
		 }
		 
         else if(dmb_tr2_check[1]){
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s %s",
	         word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],sign1,dmb_common,dmb_tr2);
		 dmb_t2_coll.push_back(word_numbering);
	         LogTrace("badData") << tempbuf1; w=0;
		 }	 
	 // TMB	 
         else if(tmb_h1_check[1]){
	         tmb_start=word_numbering;
	         tmb_inst_l1a=(buf[i+2]&0x000F);
		 tmb_l1a_coll.push_back(tmb_inst_l1a);
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s %s%s%s %i",
	         word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],sign1,tmb_common,tmb_header1,
		 sign1,dmb_common_l1a,tmb_inst_l1a);
		 tmb_h1_coll.push_back(word_numbering);
	         LogTrace("badData") << tempbuf1; w=0; tmb_inst_l1a=0; 
		 }
	 else if(tmb_tr1_check[1]){
	         tmb_stop=word_numbering;
	         if((tmb_start!=0)&&(tmb_stop!=0)&&(tmb_stop>tmb_start)) {
		        tmb_inst_wcnt2=4*(tmb_stop-tmb_start+1);
		        tmb_wcnt2_coll.push_back(tmb_inst_wcnt2);
			}
		 tmb_inst_wcnt1=(buf[i+3]&0x7FF);
		 tmb_wcnt1_coll.push_back(tmb_inst_wcnt1);
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s %s%s%s %i %s %i",
	         word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],
		 sign1,tmb_common,tmb_tr1,sign1,alct_common_wcnt1,tmb_inst_wcnt1,
		 alct_common_wcnt2,tmb_inst_wcnt2);
		 tmb_t1_coll.push_back(word_numbering);
	         LogTrace("badData") << tempbuf1; w=0;
		 tmb_inst_wcnt2=0;
		 }
	 // CFEB
	 else if(cfeb_tr1_check[1]){
	         ++cfeb_sample;
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s %s%s %s %i",
	         word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],
		 sign1,cfeb_common,cfeb_tr1,sign1,cfeb_common_sample,cfeb_sample);
		 cfeb_t1_coll.push_back(word_numbering); w=0;
	         LogTrace("badData") << tempbuf1; w=0;
		 }
	else if(cfeb_b_check[1]){
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s %s",
	         word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],sign1,cfeb_common,cfeb_b);
	         LogTrace("badData") << tempbuf1; w=0;
		 }	 	  
	//ERRORS ddu_tr1_bad_check	 
	else if(ddu_tr1_bad_check[1]){
	         ddu_inst_i = ddu_h1_n_coll.size(); ddu_inst_n=ddu_h1_n_coll[ddu_inst_i-1];
		 sprintf(tempbuf1,"%6i    %04x %04x %04x %04x%s%s%i %s %s",
	         word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i],sign1,ddu_common,ddu_inst_n,
		 ddu_trail1,ddu_tr1_err_common);
	         LogTrace("badData") << tempbuf1; w=0;
		 }	  
	
	 else if(extraction&&(!ddu_h1_check)){
	      if(w<3){
	      sprintf(tempbuf,"%6i    %04x %04x %04x %04x",word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i]);
	      LogTrace("badData") << tempbuf; w++;}
	      if(w==3){
	      LogTrace("badData") << "..................................................."; w++;}
	  }	  
	  	  
	  else if((!ddu_h1_check)){   
	  sprintf(tempbuf,"%6i    %04x %04x %04x %04x",word_numbering,buf[i+3],buf[i+2],buf[i+1],buf[i]);
	  LogTrace("badData") << tempbuf;
	  }
	  
	  i+=3; ddu_h1_check=false; 
      }
      
        char sign[30]; 
	LogTrace("badData") <<"********************************************************************************" <<
	 std::endl;
	if(fedshort)
	LogTrace("badData") << "For complete output turn off VisualFEDShort in muonCSCDigis configuration file.";
	LogTrace("badData") <<"********************************************************************************" <<
	 std::endl;
	LogTrace("badData") << std::endl; 
	LogTrace("badData") <<"            Summary                ";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << ddu_h1_coll.size() <<"  "<< ddu_common << "  "<<ddu_header1 << "  "<< "found";
	for(unsigned int k=0; k<ddu_h1_coll.size();++k){
	   sprintf(sign,"%s%6i%5s %s%i %s %i","Line: ",
	   ddu_h1_coll[k],sign1,ddu_common,ddu_h1_n_coll[k],dmb_common_l1a,ddu_l1a_coll[k]);
	   LogTrace("badData") << sign;
	}		       
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << ddu_h2_coll.size() <<"  "<< ddu_common << "  "<<ddu_header2 << "  "<< "found";
	for(unsigned int k=0; k<ddu_h2_coll.size();++k)
	   LogTrace("badData") << "Line:  " << ddu_h2_coll[k];
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << ddu_h3_coll.size() <<"  "<< ddu_common << "  "<<ddu_header3 << "  "<< "found";
	for(unsigned int k=0; k<ddu_h3_coll.size();++k)
	   LogTrace("badData") << "Line:  " << ddu_h3_coll[k];
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << ddu_t1_coll.size() <<"  "<< ddu_common << "  "<<ddu_trail1 << "  "<< "found";
	for(unsigned int k=0; k<ddu_t1_coll.size();++k)
	   LogTrace("badData") << "Line:  " << ddu_t1_coll[k];
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << ddu_t2_coll.size() <<"  "<< ddu_common << "  "<<ddu_trail2 << "  "<< "found";
	for(unsigned int k=0; k<ddu_t2_coll.size();++k)
	   LogTrace("badData") << "Line:  " << ddu_t2_coll[k];
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << ddu_t3_coll.size() <<"  "<< ddu_common << "  "<<ddu_trail3 << "  "<< "found";
	for(unsigned int k=0; k<ddu_t3_coll.size();++k)
	   LogTrace("badData") << "Line:  " << ddu_t3_coll[k];
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << dmb_h1_coll.size() <<"  "<< dmb_common << "  "<<dmb_header1 << "  "<< "found";
	for(unsigned int k=0; k<dmb_h1_coll.size();++k){
	   sprintf(sign,"%s%6i%5s %s %s %i %s %i %s %i","Line: ",
	   dmb_h1_coll[k],sign1,dmb_common,dmb_common_crate,dmb_crate_coll[k],dmb_common_slot,
	   dmb_slot_coll[k],dmb_common_l1a,dmb_l1a_coll[k]);
	   LogTrace("badData") << sign;
	   }
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << dmb_h2_coll.size() <<"  "<< dmb_common << "  "<<dmb_header2 << "  "<< "found";
	for(unsigned int k=0; k<dmb_h2_coll.size();++k)
	   LogTrace("badData") << "Line:  " << dmb_h2_coll[k];
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << dmb_t1_coll.size() <<"  "<< dmb_common << "  "<<dmb_tr1 << "  "<< "found";
	for(unsigned int k=0; k<dmb_t1_coll.size();++k)
	   LogTrace("badData") << "Line:  " << dmb_t1_coll[k];
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << dmb_t2_coll.size() <<"  "<< dmb_common << "  "<<dmb_tr2 << "  "<< "found";
	for(unsigned int k=0; k<dmb_t2_coll.size();++k)
	   LogTrace("badData") << "Line:  " << dmb_t2_coll[k];
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << alct_h1_coll.size() <<"  "<< alct_common << "  "<<alct_header1 << "  "<< "found";
	for(unsigned int k=0; k<alct_h1_coll.size();++k){
	   sprintf(sign,"%s%6i%5s %s %s %i","Line: ",
	   alct_h1_coll[k],sign1,alct_common,
	   dmb_common_l1a,alct_l1a_coll[k]);
	   LogTrace("badData") << sign;
	   }
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << alct_h2_coll.size() <<"  "<< alct_common << "  "<<alct_header2 << "  "<< "found";
	for(unsigned int k=0; k<alct_h2_coll.size();++k){
	   sprintf(sign,"%s%6i%5s %s %s %i","Line: ",
	   alct_h1_coll[k],sign1,alct_common,
	   alct_common_bxn,alct_bxn_coll[k]);
	   LogTrace("badData") << sign;
	   }
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << alct_t1_coll.size() <<"  "<< alct_common << "  "<<alct_tr1 << "  "<< "found";
	for(unsigned int k=0; k<alct_t1_coll.size();++k){
	        sprintf(sign,"%s%6i%5s %s %s %i %s %i","Line: ",
	        alct_t1_coll[k],sign1,alct_common,
	        alct_common_wcnt1,alct_wcnt1_coll[k],alct_common_wcnt2,alct_wcnt2_coll[k]);
	                        
	   LogTrace("badData") << sign;
	   }
	   
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << tmb_h1_coll.size() <<"  "<< tmb_common << "  "<<tmb_header1 << "  "<< "found";
	for(unsigned int k=0; k<tmb_h1_coll.size();++k){
	   sprintf(sign,"%s%6i%5s %s %s %i","Line: ",
	   tmb_h1_coll[k],sign1,tmb_common,
	   dmb_common_l1a,tmb_l1a_coll[k]);
	   LogTrace("badData") << sign;
	}   
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << tmb_t1_coll.size() <<"  "<< tmb_common << "  "<<tmb_tr1 << "  "<< "found";
	for(unsigned int k=0; k<tmb_t1_coll.size();++k){
	        sprintf(sign,"%s%6i%5s %s %s %i %s %i","Line: ",
	        tmb_t1_coll[k],sign1,tmb_common,
	        alct_common_wcnt1,tmb_wcnt1_coll[k],alct_common_wcnt2,tmb_wcnt2_coll[k]);
	                        
	   LogTrace("badData") << sign;
	   }
	LogTrace("badData") << std::endl;
	LogTrace("badData") << "||||||||||||||||||||";
	LogTrace("badData") << std::endl;
	LogTrace("badData") << cfeb_t1_coll.size() <<"  "<< cfeb_common << "  "<<cfeb_tr1 << "  "<< "found";
	for(unsigned int k=0; k<cfeb_t1_coll.size();++k)
	   LogTrace("badData") << "Line:  " << cfeb_t1_coll[k];
	 LogTrace("badData") <<"********************************************************************************";
	
}
