using Test
using RootCauseDiscovery

@testset "download and preprocess data" begin
    # download data
    dir = RootCauseDiscovery.datadir()
    download_data(dir)

    # process data
    genecounts = process_data(dir)
    @test size(genecounts) == (62492, 424)
    root_cause_df = process_root_cause_truth(genecounts, dir)
    @test size(root_cause_df) == (32, 5)

    # run QC
    genecounts_normalized_ground_truth,
    genecounts_normalized_obs,
        root_cause_df_new = QC_gene_expression_data()
    
end

