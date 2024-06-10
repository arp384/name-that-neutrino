import numpy as np
import sys
sys.path.append('/home/aphillips/name-that-neutrino/ntn')
import os
import subprocess
import argparse
import csv
from tables import *
import pandas as pd
from matplotlib import pyplot as plt
import h5py
import glob
from icecube.icetray import I3Units
import icecube.MuonGun 
from icecube import dataio, dataclasses, icetray, MuonGun 
from I3Tray import *
from icecube.hdfwriter import I3HDFWriter
from icecube.dataclasses import I3Particle
from I3Tray import I3Units

from enums import (
    cascade_interactions,
    classification,
    class_mapping,
    containments_types,
    interaction_types,
    nugen_int_t_mapping,
    track_interactions,
    tau_interactions,
)

try:
    from icecube import LeptonInjector # type: ignore

    LI_FOUND = True
except:
    LI_FOUND = False


# Convenience collections
negative_charged_leptons = [I3Particle.EMinus, I3Particle.MuMinus, I3Particle.TauMinus]
positive_charged_leptons = [I3Particle.EPlus, I3Particle.MuPlus, I3Particle.TauPlus]
all_charged_leptons = negative_charged_leptons + positive_charged_leptons

muon_types = [I3Particle.MuMinus, I3Particle.MuPlus]
tau_types = [I3Particle.TauMinus, I3Particle.TauPlus]
electron_types = [I3Particle.EMinus, I3Particle.EPlus]

neutrino_types = [I3Particle.NuE, I3Particle.NuMu, I3Particle.NuTau]
anti_neutrino_types = [I3Particle.NuEBar, I3Particle.NuMuBar, I3Particle.NuTauBar]
all_neutrinos = neutrino_types + anti_neutrino_types

electron_neutrinos = [I3Particle.NuE, I3Particle.NuEBar]
muon_neutrinos = [I3Particle.NuMu, I3Particle.NuMuBar]
tau_neutrinos = [I3Particle.NuTau, I3Particle.NuTauBar]

# from clsim
cascade_types = [
    I3Particle.Neutron,
    I3Particle.Hadrons,
    I3Particle.Pi0,
    I3Particle.PiPlus,
    I3Particle.PiMinus,
    I3Particle.K0_Long,
    I3Particle.KPlus,
    I3Particle.KMinus,
    I3Particle.PPlus,
    I3Particle.PMinus,
    I3Particle.K0_Short,
    I3Particle.EMinus,
    I3Particle.EPlus,
    I3Particle.Gamma,
    I3Particle.Brems,
    I3Particle.DeltaE,
    I3Particle.PairProd,
    I3Particle.NuclInt,
]




'''
Filename: APMCLabeler.py
Author: Theo Glauch, heavily modified by Andrew Phillips
Date: 6/10/24
Purpose: Combined NuGen and Corsika labelers into a single labeler, base upon
ratio of signal and background charge to total

'''

class APMCLabeler(icetray.I3Module):

    def __init__(self, context):
        super().__init__(context)
        gcd = '/home/aphillips/ntn/GeoCalibDetectorStatus_2012.56063_V1.i3.gz'
        open_gcd = dataio.I3File(gcd)
        open_gcd.rewind()
        frame1 = open_gcd.pop_frame(icetray.I3Frame.Geometry)
        i3geo = frame1['I3Geometry']

        self.AddParameter("gcd", "Path of GCD File. If none use g frame", i3geo
)
        self.AddParameter(
            "cr_muon_padding",
            "Padding for CR muons. Increase to count muons passing further out. ",
            150 * I3Units.m,
        )
        self.AddParameter(
            "det_hull_padding",
            "Padding for the detector hull for calculating containment.",
            0 * I3Units.m,
        )
        self.AddParameter(
            "mcpe_pid_map_name",
            "Name of the I3MCPESeriesMapParticleIDMap. Set to `None` to disable background MCPE counting."
            "Note: Naive MCPE downsampling will render I3MCPESeriesMapParticleIDMap useles",
            "I3MCPESeriesMapParticleIDMap",
        )
        self.AddParameter(
            "mcpe_map_name",
            "Name of the I3MCPESeriesMap",
            "I3MCPulseSeriesMap",
        )
        
        self.AddParameter(
            "mc_pulse_pid_map_name",
            "Name of the I3MCPESeriesMapParticleIDMap. Set to `None` to disable background MCPE counting."
            "Note: Naive MCPE downsampling will render I3MCPESeriesMapParticleIDMap useles",
            "I3MCPulseSeriesMapParticleIDMap", #(AP edit 2-21-24)
        )
        self.AddParameter(
            "mc_pulse_map_name",
            "Name of the I3MCPulseSeriesMap",
            "I3MCPulseSeriesMap",
        )
        
        
        self.AddParameter("mctree_name", "Name of the I3MCTree", "SignalI3MCTree")
        self.AddParameter(
            "bg_mctree_name",
            "Name of the background I3MCTree. (Change if coincident events are in a"
            " separate MCTree)",
            "I3MCTree",
        )
        self.AddParameter(
            "event_properties_name", "Name of the LI EventProperties.", None
        )
        self.AddParameter("weight_dict_name", "Name of the I3MCWeightDict", 'I3MCWeightDict')
        self.AddParameter(
            "corsika_weight_map_name", "Name of the CorsikaWeightMap", None
        )
        self.AddParameter("key_postfix", "Postfix for the keys stored in the frame", "")

        self._surface = None
        self._surface_cr = None

    def Configure(self):
        self._geo = self.GetParameter("gcd")
        self._cr_muon_padding = self.GetParameter("cr_muon_padding")
        self._det_hull_padding = self.GetParameter("det_hull_padding")
        self._mcpe_pid_map_name = self.GetParameter("mcpe_pid_map_name")
        self._mcpe_map_name = self.GetParameter("mcpe_map_name")
        self._mc_pulse_pid_map_name = self.GetParameter("mc_pulse_pid_map_name")
        self._mc_pulse_map_name = self.GetParameter("mc_pulse_map_name")
        self._mctree_name = self.GetParameter("mctree_name")
        self._bg_mctree_name = self.GetParameter("bg_mctree_name")
        self._event_properties_name = self.GetParameter("event_properties_name")
        self._weight_dict_name = self.GetParameter("weight_dict_name")
        self._corsika_weight_map_name = self.GetParameter("corsika_weight_map_name")
        self._key_postfix = self.GetParameter("key_postfix")

        if (self._event_properties_name is not None) + (
            self._weight_dict_name is not None) + (self._corsika_weight_map_name is not None) != 1:
            raise RuntimeError(
                "Set only one of event_properties_name, weight_dict_name and corsika_weight_map_name"
            )

        self._is_li = self._event_properties_name is not None
        if self._is_li and not LI_FOUND:
            raise RuntimeError(
                "Simulation is LeptonInjector but couldn't import LeptonInjector."
            )
        self._is_corsika = self._corsika_weight_map_name is not None

    def Geometry(self, frame):
        if self._geo is None:
            self._geo = frame["I3Geometry"]
        self.PushFrame(frame)

    def getSubtreeCharge(self, frame, tree, root):

        '''
        Function for computing charge contribution based on subtree of a particle
        '''

        #get a list of all the children of the particle
        #do a dfs of the tree, add pids to list each time we pop a particle
        queue = [root]
        pids = []

        while True:
            if queue == []:
                break
            p = queue.pop()
            pids.append(p.id)
            children = tree.children(p)
            for ch in children:
                queue.append(ch)
        
        mc_pulse_series_map = frame[self._mc_pulse_map_name]

        #issue: sometimes double counting pulses. Need a way to make sure we get set of unique pulses.
        all_pulses = []
        for omkey, idmap in frame[self._mc_pulse_pid_map_name]: #loop over all the omkeys in pid map
            
            mc_pulse_series = mc_pulse_series_map[omkey]        #get the pulse series on that DOM

            #id map maps pids to what pulses it's responsible for on that DOM
            indices = list(range(0, len(mc_pulse_series))) #list of indices
            for pid in idmap.keys():                            #loop over all the pids in the pid map
                
                if pid in pids: #check if the pid is in our list of children

                    #every time we see a new index pop it from the list of indices
                    for i in idmap[pid]:
                        if i in indices:
                            all_pulses.append(mc_pulse_series[i])
                            indices.remove(i)

        #now we have a list of unique pulses attributable to subtree of root particle. sum their charge to get total charge that the 
        #root is "responsible for"
        charge = sum([p.charge for p in all_pulses])

        return charge

    def getQtot(self, frame):

        '''
        Function for computing total charge in event
        
        '''
        pulse_map = frame['I3MCPulseSeriesMap'] #dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'InIceDSTPulses') #Get pulse map
        
        all_pulses = [p for i,j in pulse_map for p in j]
        return sum([p.charge for p in all_pulses])


    @staticmethod
    def find_neutrinos(tree):
        return tree.get_filter(
            lambda p: (p.type in all_neutrinos) and np.isfinite(p.length)
        )

    @staticmethod
    def get_inice_neutrino(tree, is_li):

        
        #print(tree.get_primaries())
        neutrino_primary = [p for p in tree.get_primaries() if p.type in all_neutrinos] #here is fix
        #print(len(neutrino_primary))
        if len(neutrino_primary) != 1:
            return 1
            #raise RuntimeError("Found more or less than one primary neutrino")
        neutrino_primary = neutrino_primary[0]

        if is_li:
            # Assume that LI only inserts the final, in-ice neutrino
            return neutrino_primary


        # Not LI. Find highest energy in-ice neutrino
        in_ice_nu = tree.get_best_filter(
            lambda p: (p.location_type == I3Particle.InIce) and
                      (p.type in all_neutrinos) and
                      tree.is_in_subtree(neutrino_primary, p),
            lambda p1, p2: p1.energy > p2.energy)


        # if not in_ice_nu:
        # For some reason, NuGen sometimes marks non-final neutrinos as in-ice.
        # As a work-around find the highest energy in-ice particle in the
        # subtree of the neutrino primary and work back from there
        in_ice = tree.get_best_filter(
            lambda p: (p.location_type == I3Particle.InIce)
            and ((p.type in all_charged_leptons) or (p.type == I3Particle.Hadrons))
            and tree.is_in_subtree(neutrino_primary, p),
            lambda p1, p2: p1.energy > p2.energy,
        )

        def parent_nu(part, tree):
            """Recursively find the parent neutrino"""
            if part.type in all_neutrinos:
                return part
            parent = tree.parent(part)
            return parent_nu(parent, tree)

        in_ice_nu = parent_nu(in_ice, tree)

        # Sanity check

        nu_children = tree.children(in_ice_nu)

        subnu = [
            p
            for p in nu_children
            if (p.type in all_neutrinos) and p.location_type == I3Particle.InIce
        ]
        if subnu and tree.children(subnu[0]):
            print("Warning: found two in-ice neutrinos, trying child particle")
            return subnu[0]

        return in_ice_nu
    
    @staticmethod
    def get_corsika_muons(tree):
        primaries = [p for p in tree.get_primaries() if p.type not in all_neutrinos]
        muons = [
            p
            for primary in primaries
            for p in tree.children(primary)
            if p.type in muon_types
        ]

        return muons
    
    @staticmethod
    def get_containment(
        p, surface, decayed_before_type=containments_types.no_intersect
    ):
        """
        Determine containment type for particle `p`.
        if `p` is a track, the `decayed_before_type` allows specifiying the
        containment type of particles that would intersect with the
        surface, but decay before entering.
        """

        intersections = surface.intersection(p.pos, p.dir)

        if not np.isfinite(intersections.first):
            return containments_types.no_intersect

        if p.is_cascade:
            if intersections.first <= 0 and intersections.second > 0:
                return containments_types.contained
            return containments_types.no_intersect

        if p.is_track:
            # Check if starting or contained
            if intersections.first <= 0 and intersections.second > 0:
                if p.length <= intersections.second:
                    return containments_types.contained
                return containments_types.starting

            # Check if throughgoing or stopping
            if intersections.first > 0 and intersections.second > 0:
                if p.length <= intersections.first:
                    return decayed_before_type
                if p.length > intersections.second:
                    return containments_types.throughgoing
                else:
                    return containments_types.stopping
        return containments_types.no_intersect
    
    @staticmethod
    def get_neutrino_interaction_type_nugen(wdict, tree):
        int_t = wdict["InteractionType"]
        nutype = wdict["InIceNeutrinoType"]
        neutrino = APMCLabeler.get_inice_neutrino(tree, is_li=False)

        if neutrino is None:
            return None

        children = tree.children(neutrino)
        if len(children) != 2:

            raise RuntimeError(
                "Neutrino interaction with more or less than two children."
            )

        if int_t != 3:
            return nugen_int_t_mapping[(nutype, int_t)]

        if int_t == 3:
            # GR.
            if (children[0].type == I3Particle.Hadrons) and (
                children[1].type == I3Particle.Hadrons
            ):
                return interaction_types.gr_hadronic
            if (children[0].type in electron_types) or (
                children[1].type in electron_types
            ):
                return interaction_types.gr_leptonic_e
            if (children[0].type in muon_types) or (children[1].type in muon_types):
                return interaction_types.gr_leptonic_mu
            if (children[0].type in tau_types) or (children[1].type in tau_types):
                return interaction_types.gr_leptonic_tau
        raise RuntimeError(
            "Unknown interaction type: {} -> {} + {} (Nugen type {})".format(
                neutrino.type, children[0].type, children[1].type, int_t
            )
        )

    def _classify_neutrinos(self, frame):

        tree = frame[self._mctree_name]
        
        in_ice_neutrino = self.get_inice_neutrino(tree, self._is_li)
        
        wdict = frame[self._weight_dict_name]
        int_t = self.get_neutrino_interaction_type_nugen(wdict, tree)
        if in_ice_neutrino is not None:

            qSignal = self.getSubtreeCharge(frame, tree, in_ice_neutrino)
            children = tree.children(in_ice_neutrino)               #save the children array, because it's used later

            # Classify everything related to muons
            if int_t in track_interactions:
                # figure out if vertex is contained
                muons = [p for p in children if p.type in muon_types]
                if len(muons) != 1:
                    raise RuntimeError(
                        "Muon interaction with not exactly one muon child"
                    )

                containment = self.get_containment(muons[0], self._surface)

            # Classify everything related to cascades

            elif int_t in cascade_interactions:
                cascades = [p for p in children if p.is_cascade]
                if not cascades:
                    raise RuntimeError(
                        "Found cascade-type interaction but no cascade children"
                    )
                # We can have more than one cascade, just check the first
                # TODO: Check whether there are any pitfalls with this approach

                containment = self.get_containment(cascades[0], self._surface)

            elif int_t in tau_interactions:
                taus = [p for p in children if p.type in tau_types]
                if len(taus) != 1:
                    raise RuntimeError("Tau interaction with not exactly one tau child")

                containment = self.get_containment(taus[0], self._surface)

                # if the tau is contained, check the tau decay
                if containment == containments_types.contained:
                    tau_children = tree.children(taus[0])
                    muons = [p for p in tau_children if p.type in muon_types]
                    if len(muons) > 0:
                        # the tau decays into a muon
                        containment = containments_types.tau_to_mu

                if containment == containments_types.no_intersect:
                    # Check the containment of the resulting muon
                    tau_muons = [
                        p for p in tree.children(taus[0]) if p.type in muon_types
                    ]
                    if len(tau_muons) > 1:
                        raise RuntimeError("Tau decay with more than one muon")
                    elif len(tau_muons) == 1:
                        # We have a muon

                        muon_containment = self.get_containment(
                            tau_muons[0], self._surface
                        )
                        containment = muon_containment

                        # Since the tau is uncontained, we label the event by the topology
                        # of the muon created in the tau decay
                        int_t = interaction_types.numu_cc

            else:
                raise RuntimeError("Unknown interaction type: {}".format(int_t))
        else:
            int_t = None
            containment = None
        return int_t, containment, qSignal

    def _classify_corsika(self, frame):
        """
        Classify corsika events.
        The code to distinguish bundles / single muons is not yet perfect. There might
        be edge cases, where a single muon accompanied by low-energy muons that stop far
        away from the detector is classified as skimming
        """

        tree = frame[self._bg_mctree_name]
        corsika_muons = self.get_corsika_muons(tree)
        bgCharge = sum([self.getSubtreeCharge(frame, tree, m) for m in corsika_muons])

        containments = [
            self.get_containment(
                muon, self._surface, decayed_before_type=containments_types.decayed
            )
            for muon in corsika_muons
        ]

        int_t = interaction_types.corsika

        # Check if we are dealing with a single muon.
        # Number of muons that would have intersected by decay before entering the detector
        num_decayed = len(
            [cont for cont in containments if cont == containments_types.decayed]
        )

        if num_decayed == len(containments) - 1:
            # all decayed except one. containment type is given by surviving muon
            not_decayed = [
                cont for cont in containments if cont != containments_types.decayed
            ][0]
            return int_t, not_decayed, bgCharge

        # at least one muon is uncontained
        if any([cont == containments_types.no_intersect for cont in containments]):
            return int_t, containments_types.no_intersect,bgCharge

        # All muons are stopping
        if all([cont == containments_types.stopping for cont in containments]):
            return int_t, containments_types.stopping_bundle, bgCharge

        # Bundle is throughgoing
        return int_t, containments_types.throughgoing_bundle, bgCharge
    
    def classify(self, frame):
        if self._mctree_name not in frame:
            raise RuntimeError("I3MCTree no found")

        if self._surface is None:
            self._surface = MuonGun.ExtrudedPolygon.from_I3Geometry(
                self._geo, self._det_hull_padding
            )
            self._surface_cr = MuonGun.ExtrudedPolygon.from_I3Geometry(
                self._geo, self._cr_muon_padding
            )

        
        int_t_cr, containment_cr,qbg = self._classify_corsika(frame) #classify corsika part
        
        int_t_ng, containment_ng,qsig = self._classify_neutrinos(frame) #classify nugen part

    
        #print(mcpe_from_muons)
        #print(mcpe_from_muons_charge)
        return (class_mapping.get((int_t_cr, containment_cr), classification.unclassified),
                class_mapping.get((int_t_ng, containment_ng), classification.unclassified),
                qbg,
                qsig)
    
    def DAQ(self, frame):
        if self._geo is None:
            raise RuntimeError("No geometry information found")
        cr_classification, ng_classification, qbg,qsig = self.classify(frame)
        qtotal = self.getQtot(frame)
        frame["classification" + self._key_postfix] = icetray.I3Int(int(ng_classification))
        frame["classification_label" + self._key_postfix] = dataclasses.I3String(
            ng_classification.name
        )
        frame["corsika_label" + self._key_postfix] = icetray.I3Int(int(cr_classification))
        frame["corsika_classification" + self._key_postfix] = dataclasses.I3String(cr_classification.name)

        frame["signal_charge" + self._key_postfix] = dataclasses.I3Double(qsig)
        frame["bg_charge" + self._key_postfix] = dataclasses.I3Double(qbg)
        frame["qtot" + self._key_postfix] = dataclasses.I3Double(qtotal)
        self.PushFrame(frame)