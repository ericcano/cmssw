import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

EMCLUEGPUProd = cms.EDProducer('HGCalLayerClusterProducerEMGPU',
                               HGCEEInputGPU = cms.InputTag('EERecHitGPUProd'),
                               dc = cms.double(1.3),
                               kappa = cms.double(9.),
                               ecut = cms.double(3.),
                               outlierDeltaFactor = cms.double(2.),
)

                               #values for dc
                               # 1.3,
                               # 1.3,
                               # 5.0,
                               # 0.0315,  // for scintillator
