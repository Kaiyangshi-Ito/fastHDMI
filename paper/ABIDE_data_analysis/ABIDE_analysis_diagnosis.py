csv_file = r"/home/kyang/projects/def-cgreenwo/abide_data/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv"
# abide = pd.read_csv(csv_file, encoding='unicode_escape', engine="c")
abide = dd.read_csv(csv_file, sample=1250000)

# _abide_name = abide.columns.tolist()[1:]
_abide_name = list(abide.columns)[1:]

# print(_abide_name)

# we don't inlcude age and sex in the screening since they should always be included in the model
abide_name = [_abide_name[-1]] + _abide_name[1:-3]
# so that the left first column is the outcome and the rest columns are areas

mi_output = mi.binary_screening_csv_parallel(csv_file,
                                             _usecols=abide_name,
                                             csv_engine="c",
                                             sample=1250000,
                                             multp=10)
np.save(r"./ABIDE_diagnosis_MI_output", mi_output)

pearson_output = mi.Pearson_screening_csv_parallel(csv_file,
                                                   _usecols=abide_name,
                                                   csv_engine="c",
                                                   sample=1250000,
                                                   multp=10)
np.save(r"./ABIDE_diagnosis_Pearson_output", pearson_output)
