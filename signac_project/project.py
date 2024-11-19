#!/opt/miniconda3/bin/python
"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python project.py --help
"""
import flow
from flow import FlowProject
import os
import subprocess
import numpy as np

def gen_seq_str(job):
    n_azo = round(int(job.sp["chain_N"])*float(job.sp["chain_frac_azo"]))
    n_PEO = int(job.sp["chain_N"]) - n_azo
    n_PEO_block = job.sp["chain_N"]/(n_azo+1)
    PEO_block_len = round(n_PEO / n_PEO_block)

    #generate the repeating part
    seq_str = '-(B'
    for i in range(PEO_block_len):
        seq_str += 'A'
    seq_str += rf')x{n_PEO_block-1}'

    #prepend the starting PEO block
    prefix = ''
    for i in range(n_PEO - PEO_block_len*(n_PEO_block-1)):
        prefix += 'A'
    seq_str = prefix + seq_str
    
    return seq_str
    
class MyProject(FlowProject):
    pass

class CampusCluster(flow.environment.DefaultSlurmEnvironment):
    hostname_pattern = r".*.campuscluster\.illinois\.edu$"
    #template = "campuscluster.sh"

# Labels for identification
#-----------

# labels you define show up at the top with progress bars when doing "python project.py status"

@MyProject.label
def initialized(job):
    return job.isfile("init.gsd")

@MyProject.label
def isomerized(job):
    if job.sp["azo_isomer"] == 'trans_to_cis':
        return job.isfile("isomerize.gsd")
    else:
        return job.isfile("init.gsd")

@MyProject.label
def existed(job):
    return job.isfile("exist.gsd")

@MyProject.label
def rdf_cat_an_analyzed(job):
    return job.isfile("plots/rdf_cation_anion.pdf")

@MyProject.label
def cluster_analyzed(job):
    return job.isfile("cluster.gsd")

# # this can be changed after some simulations are run, and if 1e5 is increased, signac will
# # know to re-run all simulations for longer
# @MyProject.label
# def equilibrated(job):
#     return job.doc['equi_step'] >= 1e5

#-----------------------------------
#   label for orientation analysis
#-----------------------------------

# Workflow - i.e actual simulations
#-----------

# if simulation is not initialized - initialize!
# operation name = function name
@MyProject.post(initialized)
@MyProject.operation
def initialize(job):
    from scripts.simulate import Simulator
    sinit = Simulator(job)
    sinit.initialize()
    print("initialized: ",job.id)

@MyProject.post(isomerized)
@MyProject.pre(initialized)
@MyProject.operation
def isomerize(job):
    from scripts.simulate import Simulator
    sinit = Simulator(job)
    sinit.isomerize()
    print("isomerized: ",job.id)

@MyProject.post(existed)
@MyProject.pre(isomerized)
@MyProject.operation
def exist(job):
    from scripts.simulate import Simulator
    sinit = Simulator(job)
    sinit.exist()
    print("existed: ",job.id)

# this workflow could have many more steps depending on how complicated the project is
# ....

# Analysis
#-----------
# operations can be added/modified after the project started

@MyProject.post(rdf_cat_an_analyzed)
@MyProject.pre(existed)
@MyProject.operation
def analyze_cat_an_rdf(job):
    from scripts.analyze import Analyzer
    analyzer = Analyzer(job)
    analyzer.plot_rdf_cation_anion()
    print("RDF analysis done for job: ",job.id)

@MyProject.post(cluster_analyzed)
@MyProject.pre(existed)
@MyProject.operation
def analyze_cluster(job):
    from scripts.analyze import Analyzer
    analyzer = Analyzer(job)
    analyzer.cluster_analysis()
    print("Cluster analysis done for job: ",job.id)

# #Draws data from the PPA analysis and also remaps the chain conformation
# #   information from the first frame of ppa_out.gsd to the normal chain
# #   representation. Produces a conformationAnalysis.gsd file to be used in
# #   other analyses
# @MyProject.post(conformationAnalyzed)
# @MyProject.pre.isfile("deform.gsd")
# @MyProject.operation
# def analyzeConformations(job):
#     from scripts.analyze import ConformationAnalyzer
#     analyzer = ConformationAnalyzer(job.fn('deform.gsd'),'conformationAnalysis')
#     if not (os.path.isfile(job.fn('ppa.gsd'))):
#         analyzer.analyzePPA(job)
#     analyzer.analyzeConformations(job)
#     print("Conformational analysis done for job: ",job.id)

# #Takes the conformationAnalysis.gsd as input and analyzes the double well bond
# #   length to determine how many are activated. Gathers these statistics and
# #   separates according to chain conformation (velocity[0]). These statistics
# #   are then returned in an output activationAnalysis.txt file, with one line
# #   per frame analyzed.
# @MyProject.post(activationAnalyzed)
# @MyProject.pre(conformationAnalyzed)
# @MyProject.operation
# def analyzeActivation(job):
#     from scripts.analyze import ActivationAnalyzer
#     analyze = ActivationAnalyzer(job.fn('conformationAnalysis.gsd'),
#                                                         "activationAnalysis",1)
#     activation =  analyze.activationAnalysis(job)
#     print("Activation analysis done for job: ",job.id)

# #Takes the conformationAnalysis.gsd as input and analyzes the backbone bond
# #   lengths of each chain, identifying if any are broken. This chain information
# #   is then kept in the particles.charge data, 0 if fine, 1 if broken. This data
# #   is consistent along the chain. This new information is stored and placed in
# #   the bondBreakingAnalysis.gsd file. The conformation analysis is required
# #   for the tabulation of statistics on chain breaking, returned in the
# #   bondBreakingAnalysis.txt file.
# @MyProject.post(bondBreakingAnalyzed)
# @MyProject.pre.isfile("conformationAnalysis.gsd")
# @MyProject.operation
# def analyzeBondBreaking(job):
#     from scripts.analyze import BondBreakingAnalyzer
#     analyzer = BondBreakingAnalyzer(job.fn('conformationAnalysis.gsd'),"bondBreakingAnalysis",period=1)
#     analyzer.analyzeBondBreaking(job)
#     print("Bond Breaking analysis done for job: ",job.id)

# this is a calculation that leads to a single numeric result for one simulation
# it is easy to save that one number in job.doc instead of making a whole file for it
# run with  "python project.py run -o analyze_num_neighbors" if there are elegible statepoints
# @MyProject.post(lambda job: "fraction_neighbors" in job.doc)
# @MyProject.pre(equilibrated)
# @MyProject.operation
# def analyze_num_neighbors(job):
#     from scripts.analyze import Analyzer
#     analyze = Analyzer(job.fn('equi.gsd'))
#     fraction_neighbors = analyze.fraction_neighbors()
#     job.doc['fraction_neighbors'] = fraction_neighbors
#     print("fraction neighbors in job: ",job.id,job.doc['fraction_neighbors'])

# this saves a file so we can use the file existence to test if this needs to be executed
# run with "python project.py run -o analyze_rdf" of there are elegible statepoints
# @MyProject.post.isfile("rdf.hist")
# @MyProject.pre(equilibrated)
# @MyProject.operation
# def analyze_rdf(job):
#     from scripts.analyze import Analyzer
#     analyze = Analyzer(job.fn('equi.gsd'))
#     rdf = analyze.rdf()
#     output_file = job.fn('rdf.hist')
#     np.savetxt(output_file,rdf,header="# bin, RDF")
#     print("analyzed rdf for job: ",job.id)


if __name__ == "__main__":
    MyProject().main()
