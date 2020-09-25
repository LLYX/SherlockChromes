#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import click
import numpy as np
import pandas as pd
import sqlite3

def import_pqp_library( con, cursor ):
    """Import only target peptide sequences (DECOY=0) from the PEPTIDE table in a PQP library database."""
    query = "SELECT PEPTIDE.MODIFIED_SEQUENCE FROM PEPTIDE WHERE PEPTIDE.DECOY=0"
    res = cursor.execute(query)
    tmp = res.fetchall()
    assert len(tmp) > 0
    return tmp

def gen_ground_truth_labels( in_chromatograms_csv, ground_truth_file, out_file ):
    """Generate a binary array of ground truth labels"""

    click.echo( "in_chromatograms_csv: {0}\nground_truth_file: {1}".format(in_chromatograms_csv,ground_truth_file))

    # Check files for supported formats
    in_chromatograms_csv_filename, in_chromatograms_csv_ext = os.path.splitext( in_chromatograms_csv )
    if  ( in_chromatograms_csv_ext != '.tsv' and in_chromatograms_csv_ext != '.csv' and in_chromatograms_csv_ext != '.txt' ):
        raise ValueError( 'chromatogram csv file of type {} is not supported. Your chromatogram csv file type should be .tsv/.csv/.txt'.format(in_chromatograms_csv_ext) )

    truth_filename, truth_ext = os.path.splitext( ground_truth_file )
    if  ( truth_ext != '.tsv' and truth_ext != '.csv' and truth_ext != '.txt' ):
        raise ValueError( 'ground truth file of type {} is not supported. Your ground truth file type should be .tsv/.csv/.txt'.format(truth_ext) )
    
    # Get file delimiters
    if in_chromatograms_csv_ext == '.tsv' or in_chromatograms_csv_ext == '.txt' :
        chrom_csv_delim = "\t"
    else:
        chrom_csv_delim = ","
    
    if truth_ext == '.tsv' or truth_ext == '.txt' :
        truth_ext_delim = "\t"
    else:
        truth_ext_delim = ","

    # Load inputs into pandas dataframe
    chromatogram_df = pd.read_csv( in_chromatograms_csv, sep=chrom_csv_delim )
    truth_df = pd.read_csv( ground_truth_file, sep=truth_ext_delim )

    # Extract Filename columna and convert to a list
    filename = chromatogram_df['Filename'].tolist()
    # Convert truth_df to list
    truth_list = truth_df.to_numpy().flatten().tolist()

    # Identify which filenames contain peptide sequences from truth list
    binary_labels = [ len(set(os.path.basename(current_filename).split("_")) & set(truth_list))>0 if  1 else 0 for current_filename in filename ]
    binary_labels = np.array( binary_labels )

    # Write labels to file
    click.echo( "Writing binary labels to disk with shape: {0}.\nOutput file: {1}".format(np.shape(binary_labels), out_file) )
    np.save( out_file, binary_labels.astype(np.int32) )

def gen_target_decoy_labels( in_chromatograms_csv, library_database_file, out_file ):
    """Generate a binary array of target-decoy labels"""

    click.echo( "in_chromatograms_csv: {0}\nlibrary_database_file: {1}".format(in_chromatograms_csv,library_database_file))

    # Check files for supported formats
    in_chromatograms_csv_filename, in_chromatograms_csv_ext = os.path.splitext( in_chromatograms_csv )
    if  ( in_chromatograms_csv_ext != '.tsv' and in_chromatograms_csv_ext != '.csv' and in_chromatograms_csv_ext != '.txt' ):
        raise ValueError( 'chromatogram csv file of type {} is not supported. Your chromatogram csv file type should be .tsv/.csv/.txt'.format(in_chromatograms_csv_ext) )

    library_database_file_filename, library_database_file_ext = os.path.splitext( library_database_file )
    if  ( library_database_file_ext != '.pqp' ):
        raise ValueError( 'library file of type {} is not supported. Your library file type should be .pqp'.format(library_database_file_ext) )
    
    # Get file delimiters
    if in_chromatograms_csv_ext == '.tsv' or in_chromatograms_csv_ext == '.txt' :
        chrom_csv_delim = "\t"
    else:
        chrom_csv_delim = ","
    
    # Load inputs into pandas dataframe
    chromatogram_df = pd.read_csv( in_chromatograms_csv, sep=chrom_csv_delim )

    # Load Library File
    con = sqlite3.connect( library_database_file )
    cursor = con.cursor()
    lib_df = import_pqp_library( con,cursor )
    con.close()
    target_sequences = [ sequence[0] for sequence in lib_df ]

    # Extract Filename column and convert to a list
    filename = chromatogram_df['Filename'].tolist()

    # Identify which filenames contain peptide sequences from truth list
    binary_labels = [ len(set(os.path.basename(current_filename).split("_")) & set(target_sequences))>0 if  1 else 0 for current_filename in filename ]
    binary_labels = np.array( binary_labels )

    # Write labels to file
    click.echo( "Writing binary labels to disk with shape: {0}.\nOutput file: {1}".format(np.shape(binary_labels), out_file) )
    np.save( out_file, binary_labels.astype(np.int32) )

@click.command()
@click.option( '-in_chrom_csv', '--in_chromatograms_csv', required=True, help='Chromatogram csv file containing filename column containing peptide sequence. This file should be derived from osw_parser.' )
@click.option( '-in_truth', '--ground_truth_file', default="", help='Single column ground truth file containing peptide sequences that you know are present. i.e. synthetic peptides.' )
@click.option( '-in_lib', '--library_database_file', default="", help='A library file that contains precursor target-decoy information. Filetype: pqp' )
@click.option( '-out_file', '--out_file', default="binary_labels.npy", help='Numpy file to write binary labels to.' )
def main( in_chromatograms_csv, ground_truth_file, library_database_file, out_file ):

    if ground_truth_file!="" and library_database_file=="":
        out_file = "ground_truth_based_" + out_file 
        gen_ground_truth_labels( in_chromatograms_csv, ground_truth_file, out_file )
    elif ground_truth_file=="" and library_database_file!="":
        out_file = "target_decoy_based_" + out_file
        gen_target_decoy_labels( in_chromatograms_csv, library_database_file, out_file )
    elif ground_truth_file=="" and library_database_file=="":
        raise ValueError( "You have supplied both a ground truth file and a library file. Only supply one or the other to generate groung truth labels or target-decoy labels." )
    else:
        raise ValueError( "You did not supply a ground truth file or a library file. You need to supply one of these files to generate binary labels." )

if __name__ == '__main__':
    main()