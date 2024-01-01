# Route-sota

The route-sota project is an implementation intended to test, examine, and measure the research results of the paper "Efficient Stochastic Routing in Path-Centric Uncertain Road Networks".

This readme is composed of 3 parts:

- Folder overview: Explains what types of files can be found where in the project.
- User guide: Gives both a quick guide and a detailed guide for how to transform raw data into routing information.
- File overview. Describes what programs in the project requires which files, and where they come from.

## Folder overview

The program is split into three important directories:

- data
  - Contains the files used to generate new files and those generated files. It is split into 3 levels of directories. 
    1. The `data/` directory contains files concerning themselves with geographic data and time measurements, such as trajectories, graph descriptions, and queries for routing.
    2. The first level of under-directories directly under `data/` contains the files generated based on chosen trajectories and geographic locations, as well as the number of threads used to generate these, as that results in different files and processing thereof.
    3. The under-directories can contain a set of one or more under-directories themselves, which files vary depending on the chosen sigma value and the chosen queries to pursue. Note that queries are used to "limit" `get_policy_u.py` to relevant vertices, not for any actual computation.
  - The actual data-files used for testing can be found in a google drive folder, for which access can be provided, if asked. Otherwise, the necessary data-files must be constructed manually, where possible.
- generate
  - Contains the programs used to generate the files necessary for the routing algorithms. Contains 5 programs of import. For a detailed explanation of their usage, see the user guide. It also contains some shell-files that can be used to run the files, but it is more useful to manually run each individual python file as a number of manual edits are needed inside each for any modifications or changes.
    - `get_overlap.py`
    - `get_policy_U.py`
    - `get_tpath.py`
    - `get_vpath.py`
    - `get_vpath_desty.py`
- routing
  - Contains the routing algorithms that the project uses. Note that it contains both the python files and shell-files to run said files. The python files require manual adjustments and modifications for any most changes, so it is advised to run the python files manually. See the paper for "Efficient Stochastic Routing in Path-Centric Uncertain Road Networks" for descriptions of what each is.
    - `T-B-E.py`
    - `T-B-EU.py`
    - `T-B-P.py`
    - `T-BS.py`
    - `T-None.py`
    - `V-B-P.py`
    - `V-BS.py`
    - `V-None.py`


The project still includes the old readmes. See the readme in each of the above three folders or the file `OLDREADME.md`.

## User guide

#### Quick guide

Requirements:

- CSV file of trajectories
  - See detailed guide
- Speedfile ala AAL_NGR
  - Unknown at time of writing how to reproduce. A file for Aalborg was contained in the original project, but its exact nature is unknown.
- Vertices.txt
  - A csv-like list of all vertices in the graph that is to be used.
  - Tab-separated file containing 3 columns
    - vertex id, longitude, latitude

- Queries.txt
  - Binary file of queries between start and end nodes. Nodes are by ID.

1. Run "get_tpath"
2. Run "get_overlap"
3. Run "get_vpath"
4. Run "get_vpath_desty"
5. Run "get_policy_U"
6. Run your routing algorithm of choice

#### Detailed guide

1. Decide on a common subpath in all the files. This is where the files that you are/will be using will be stored. 

2. Download trajectory dataset in the form of a .csv file

   1. Code demands a certain schema for the data

      1. | Column name | Type                        |
         | ----------- | --------------------------- |
         | <u>id</u>   | bigint                      |
         | ntrip_id    | bigint                      |
         | trip_id     | bigint                      |
         | start_time  | timestamp without time zone |
         | end_time    | timestamp without time zone |
         | vehicle_id  | integer                     |
         | seg_id      | text                        |
         | seq_id      | integer                     |
         | travel_time | integer                     |

         

      2. The table "trips_short" on the "au_db" databse nearly has the same schema, but has since the original creation of the program been extended with two additional columns, which must be excluded in order to use it, as they will cause the program to fail if included.

3. Download and structure vertices.txt file

   1. format of file should be a tab-separeted file with 3 "columns"
      1. First column is vertex id
      2. Second is longitude
      3. Third is latitude

   2. The table "vertices" in the "au_db" database has the correct schema. For a quick file-setup, query the data from the db, and then do a dirty find-and-replace all `,` with `\t`

4. You must also have what the program calls a "speedfile" or "the city map". The file included in the original test data is AAL_NGR. This file is not produced by the program. Its exact construction and the details about its content are currently unknown.

5. Running "get_tpath.py"

   1. Remember to set your subpath on line 323.

   2. Remember to set the file name of your CSV file on line 324.

   3. Set the number of threads (process_num) to use on line 322.

      1. Note this number down. It will be used later again.

   4. Optionally, you can modify the naming of generated files by changing the dinx variable on line 321.

      1. If changed, note down change for later reuse.
      1. Best practice seems to be to keep dinx same as process_num

   5. Run the program

      1. You will now have

      | File                     |
      | ------------------------ |
      | [Trajectory].csv         |
      | path_travel_time_%d.json |

      

6. Running "get_overlap.py"

   1. Set your subpath again on line 192

   2. Set "filename" variable on line 193 to be that of your trajectory CSV file.

   3. Set the name of your count file on line 195

      1. This is a new file, and can be freely named.
         1. If changed, note down name for further usage.

   4. Set the name of your desty file on line 196

      1. This is a new file, and can be freely named.
         1. If changed, note down name for further usage.

   5. Set the name of your overlap file on line 197

      1. This is a new file, and can be freely named.
         1. If changed, note down name for further usage.

   6. Adjust the subpath_range on line 194.

      1. This should be the interval of generated path_travel_time_%d.json files that was generated in Step 5. If you had a thread count of 10, then that would be an interval [2;12]

   7. Set your dinx in line 191 to be the same as the one in Step 5.

   8. Run the program

      1. You will now have

         | File                     |
         | ------------------------ |
         | [Trajectory].csv         |
         | path_travel_time_%d.json |
         | overlap.txt              |
         | path_count.json          |
         | path_desty.json          |

7. Running "get_vpath"

   1. Set your dinx on line 474

      1. This should be the same as in steps 5 and 6

   2. Set your subpath on line 475

   3. Set the path for your speed/graph/map file (NGR) on line 476

   4. Set the location of your trajectory CSV file on line 477

   5. Set the name of your count file on line 479

      1. This was the name chosen in step 6

   6. Set the name of your desty file on line 480

      1. This was the name chosen in step 6

   7. Set the name of your overlap file on line 481

      1. This was the name chosen in step 6

   8. Set the path of your vertices file on line 482

      1. This was the one downloaded from the database

   9. Adjust your subpath_range on line 483

      1. This should be the interval of generated path_travel_time_%d.json files that was generated in Step 3. If you had a thread count of 10, then that would be an interval [2;12]

   10. Set your threads_num on line 485. Properly best to be the same as the one that has been used throughout the generation phase.

   11. Set the name of your future KKdesty_num and KKgraph files on lines  486 and 489, respectively.

       1. These are new files, and can be freely named.
          1. If changed, note down name for further usage.
       2. The degree files `KKdegree_%d.json` and `KKdegree2_%d.json`are currently not used, and can be skipped.

   12. Note that some magic-number `B` is also used for the generation, but the numbers exact meaning is unknown.

   13. Run the program

       1. You will now have

          | File                     |
          | ------------------------ |
          | [Trajectory].csv         |
          | path_travel_time_%d.json |
          | overlap.txt              |
          | path_count.json          |
          | path_desty.json          |
          | KKdesty_num_%d.json      |
          | KKgraph_%d.json          |

          

8. Running "get_vpath_desty.py"

   1. Set your dinx on line 35

   2. Set the path for your trajectory CSV file in line 36

   3. There is the same magic number `B` on line 37 as in step 7. Keep it the same as there.

   4. Set the name of your M_edge_desty file on line 66

      1. This is a new file, and can be freely named.
         1. If changed, note down the name for further usage.

   5. Run the program

      1. You will now have

      | File                     |
      | ------------------------ |
      | [Trajectory].csv         |
      | path_travel_time_%d.json |
      | overlap.txt              |
      | path_count.json          |
      | path_desty.json          |
      | KKdesty_num_%d.json      |
      | KKgraph_%d.json          |
      | M_edge_desty.json        |

9. Running "get_policy_U.py"

   1. program uses a pair of sigma and eta values. These can be set by either running the program with the argument ``--sig %d`` - where %d is [0-5] - or by manually changing the default argument on line 313.
      1. The exact relationship between sigma and eta is unknown. The current values, outside those set by the original developer, were found by plotting the values in a graph, and performing regression to compute a function `f(sigma) = eta`.

   2. Set the number of threads to run (process_num) on line 331
      1. Should be the same as in previous steps

   3. Set the time_budget on line 332
      1. This is a new value. Note it down for reuse in routing.

   4. Set the dinx on line 334. Should be set to the same as in prior steps
   5. Set your subpath on line 335
   6. Set the file to your .csv file on line 336.
   7. Set the path to your `KK_desty_num_%d.json` file on line 337
   8. Set the path to your `KKgraph_%d.json` file on line 338
   9. Set the path to your `M_edge_desty.json` file on line 339
   10. Set the path for your umatrix_path on line 340
       1. This is a new path, and can be freely named.
          1. If changed, note down the name for further usage.
       2. It should be in your subpath.

   11. Set the path for your speed/graph/map file(AAL_NGR) on line 342.
   12. Set the path for your query file on line 343.
   13. Run the program
       1. You should now have:


| File                                      |
| ----------------------------------------- |
| [Trajectory].csv                          |
| path_travel_time_%d.json                  |
| overlap.txt                               |
| path_count.json                           |
| path_desty.json                           |
| KKdesty_num_%d.json                       |
| KKgraph_%d.json                           |
| M_edge_desty.json                         |
| u_mul_matrix_sig%d/ (this is a directory) |

You now have All the files you need to run the routing algorithms. Before running one, remember to set all the paths and values to those that you desire. For an example:

##### Detailed instruction on using routing algorithms.

The following section will describe in detail how to adjust and customize the individual routing algorithms. Note that the specific line numbers might have been shuffled around a bit. All values to adjust can be found in the bottom of the algorithm's file.

Some of these files contain argument parsers to handle program arguments, but the programs are currently not set up to handle this, and they are therefore obsolete.

All of the programs should print out a detailed error message, if any ever occur in the program, which should help with identifying the location of the error.

###### T-B-E

Note that the class definition for Rout in the top of the file has three values B, pairs_num, and speed that are set by the constructor.

1. set your sigma and eta values. An interval is already set up, from which choices can be made. They appear to follow some pattern. Either run the program with argument `--sig %d`, where %d is [0,5], or change the default value on line 430
2. Set your threads_num on line 444
3. Set your dinx on line 445
4. Set your subpath on line 447
5. Set your KKdesty/fpath_desty on line 448
6. Set your M_edge_desty/fedge_desty on line 449
7. Set your vertices file/axes_file on line 450
8. Set your speed/graph/map(AAL_NGR) file on line 451
9. Set your queries file on line 452
10. Set your time budget on line 454.
11. Run the routing algorithm, and copy the results from the terminal into where you wish to store the results.

###### T-B-EU

Note that the class definition for Rout in the top of the file has three values B, pairs_num, and speed that are set by the constructor.

1. set your sigma and eta values. An interval is already set up, from which choices can be made. They appear to follow some pattern. Either run the program with argument `--sig %d`, where %d is [0,5], or change the default value on line 397
2. Set your threads_num on line 409
3. Set your dinx on line 410
4. Set your subpath on line 412
5. Set your KKdesty/fpath_desty on line 413
6. Set your M_edge_desty/fedge_desty on line 414
7. Set your vertices file/axes_file on line 415
8. Set your speed/graph/map(AAL_NGR) file on line 416
9. Set your queries file on line 417
10. Set your time budget on line 424.
11. Run the routing algorithm, and copy the results from the terminal into where you wish to store the results.

###### T-B-P

Note that the class definition for Modified in the top of the file has three values B, pairs_num, and speed that are set by the constructor.

1. set your sigma and eta values. An interval is already set up, from which choices can be made. They appear to follow some pattern. Either run the program with argument `--sig %d`, where %d is [0,5], or change the default value on line 510
2. Set your threads_num on line 526
3. Set your dinx on line 527
4. Set your subpath on line 529
5. Set your true_path/path_desty%d.json on line 530
6. Set your KKdesty/fpath_desty on line 531
7. Set your M_edge_desty/fedge_desty on line 532
8. Set your vertices file/axes_file on line 433
9. Set your speed/graph/map(AAL_NGR) file on line 534
10. Set your queries file on line 535
11. Set your time budget on line 548
12. Run the routing algorithm, and copy the results from the terminal into where you wish to store the results.

###### T-BS

Note that the class definition for Routin the top of the file has two values B and pairs_num that are set by the constructor.

1. set your sigma and eta values. An interval is already set up, from which choices can be made. They appear to follow some pattern. Either run the program with argument `--sig %d`, where %d is [0,5], or change the default value on line 440
2. Set your threads_num on line 458
3. Set your dinx on line 459
4. Set your subpath on line 461
5. Set your KKdesty/fpath_desty on line 462
6. Set your M_edge_desty/fedge_desty on line 463
7. Set your vertices file/axes_file on line 464
8. Set your speed/graph/map(AAL_NGR) file on line 465
9. Set your queries file on line 466
10. Set your fpath/u_mul_matrix_sig%d/ path on line 494
11. Set your time budget on line 596
12. Run the routing algorithm, and copy the results from the terminal into where you wish to store the results.

###### T-None

Note that the class definition for PaceRout in the top of the file has three values B, pairs_num, and speed that are set by the constructor.

1. set your sigma and eta values. An interval is already set up, from which choices can be made. They appear to follow some pattern. Either run the program with argument `--sig %d`, where %d is [0,5], or change the default value on line 474
2. Set your threads_num on line 488
3. Set your dinx on line 489
4. Set your subpath on line 491
5. Set your KKdesty/fpath_desty on line 492
6. Set your M_edge_desty/fedge_desty on line 493
7. Set your vertices file/axes_file on line 494
8. Set your speed/graph/map(AAL_NGR) file on line 495
9. Set your queries file on line 496
10. Set your fpath/u_mul_matrix_sig%d/ path on line 497
11. Set your time budget on line 499
12. Run the routing algorithm, and copy the results from the terminal into where you wish to store the results.

###### V-B-P

Note that the class definition for Modified in the top of the file has three values B, pairs_num, and speed that are set by the constructor.

1. set your sigma and eta values. An interval is already set up, from which choices can be made. They appear to follow some pattern. Either run the program with argument `--sig %d`, where %d is [0,5], or change the default value on line 563
2. Set your threads_num on line 577
3. Set your dinx on line 578
4. Set your subpath on line 580
5. Set your true_path/path_desty%d.json on line 581
6. Set your KKdesty/fpath_desty on line 582
7. Set your M_edge_desty/fedge_desty on line 583
8. Set your vertices file/axes_file on line 584
9. Set your speed/graph/map(AAL_NGR) file on line 585
10. Set your queries file on line 586
11. Set your time budget on line 588
12. Run the routing algorithm, and copy the results from the terminal into where you wish to store the results.

###### V-BS

Note that the class definition for Rout in the top of the file has two values B and pairs_num that are set by the constructor.

1. set your sigma and eta values. An interval is already set up, from which choices can be made. They appear to follow some pattern. Either run the program with argument `--sig %d`, where %d is [0,5], or change the default value on line 462
2. Set your threads_num on line 480
3. Set your dinx on line 481
4. Set your subpath on line 483
5. Set your KKdesty/fpath_desty on line 484
6. Set your M_edge_desty/fedge_desty on line 485
7. Set your vertices file/axes_file on line 486
8. Set your speed/graph/map(AAL_NGR) file on line 487
9. Set your queries file on line 488
10. Set your fpath/u_mul_matrix_sig%d on line 489
11. Set your time budget on line 591
12. Run the routing algorithm, and copy the results from the terminal into where you wish to store the results.

###### V-None

Note that the class definition for PaceRout in the top of the file has three values B, pairs_num, and speed that are set by the constructor.

1. set your sigma and eta values. An interval is already set up, from which choices can be made. They appear to follow some pattern. Either run the program with argument `--sig %d`, where %d is [0,5], or change the default value on line 439
2. Set your threads_num on line 453
3. Set your dinx on line 454
4. Set your subpath on line 456
5. Set your true_path/path_desty%d.json on line 457
6. Set your KKdesty/fpath_desty on line 458
7. Set your M_edge_desty/fedge_desty on line 459
8. Set your vertices file/axes_file on line 460
9. Set your speed/graph/map(AAL_NGR) file on line 461
10. Set your queries file on line 462
11. Set your fpath/u_mul_matrix_sig%d on line 463
12. Set your time budget on line 466
13. Run the routing algorithm, and copy the results from the terminal into where you wish to store the results.

## file overview

Below table shows an overview of what files are produced and needed for what generator programs.

| Program         | Requirements                                                 | Produces                                                     |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| get_vpath_desty | *.csv<br />path_travel_time_%d.json<br />path_desty%d.json   | M_edge_desty.json                                            |
| get_vpath       | AAL_NGR <br />*.csv <br />path_desty%d.json <br />path_count%d.json <br />overlap%d.txt <br />vertices.txt | KKdesty_num\_%d.json<br />KKgraph_%d.txt                     |
| get_overlap     | *.csv<br />path_travel_time_%d.json<br />                    | path_count%d.json<br />path_desty%d.json<br />overlap%d.json |
| get_tpath       | *.csv                                                        | path_travel_time_%d.json                                     |
| get_policy_U    | *.csv<br />KKdesty_num\_%d.json<br />M_edge_desty.json<br />KKgraph_%d.json<br />AAL_NGR<br />queries.txt | u_mul_matrix_sig%d/<br /><t>%d                               |

The below table shows which files are necessary for what routing algorithms. The sigma, eta, and time_budget values are used to adjust some of the algorithms, as can be read about in the paper "Efficient Stochastic Routing in Path-Centric Uncertain Road Networks". Some of the possible eta values programmed into the code are found using an approximate function, and can therefore be invalid, but that is unknown at time of writing.

| Program/Algorithm | Requirements                                                 |
| ----------------- | ------------------------------------------------------------ |
| T-B-E             | The `sigma` value<br />The `threads_num` value<br />The `dinx` value<br />The `time_budget`<br />`kkdesty_num_%d.json`<br />`M_edge_desty.json`<br />`vertices.txt`<br />`AAL_NGR`<br />`queries.txt` |
| T-B-EU            | The `sigma` value<br />The `threads_num` value<br />The `dinx` value<br />The `time_budget`<br />`kkdesty_num_%d.json`<br />`KKgraph_%d.txt`<br />`M_edge_desty.json`<br />`vertices.txt`<br />`AAL_NGR`<br />`queries.txt` |
| T-B-P             | The `sigma` value<br />The `threads_num` value<br />The `dinx` value<br />The `time_budget`<br />`kkdesty_num_%d.json`<br />`path_desty%d.json`<br />`M_edge_desty.json`<br />`vertices.txt`<br />`AAL_NGR`<br />`queries.txt` |
| T-BS              | The `sigma` value<br />The `eta` value<br />The `threads_num` value<br />The `dinx` value<br />The `time_budget`<br />`kkdesty_num_%d.json`<br />`M_edge_desty.json`<br />`vertices.txt`<br />`AAL_NGR`<br />`queries.txt`<br />The `u_mul_matrix_sig%d/` folder |
| T-None            | The `sigma` value<br />The `eta` value<br />The `threads_num` value<br />The `dinx` value<br />The `time_budget`<br />`kkdesty_num_%d.json`<br />`M_edge_desty.json`<br />`vertices.txt`<br />`AAL_NGR`<br />`queries.txt`<br />The `u_mul_matrix_sig%d/` folder |
| V-B-P             | The `sigma` value<br />The `threads_num` value<br />The `dinx` value<br />The `time_budget`<br />`kkdesty_num_%d.json`<br />`M_edge_desty.json`<br />`vertices.txt`<br />`AAL_NGR`<br />`queries.txt` |
| V-BS              | The `sigma` value<br />The `eta` value<br />The `threads_num` value<br />The `dinx` value<br />The `time_budget`<br />`kkdesty_num_%d.json`<br />`M_edge_desty.json`<br />`vertices.txt`<br />`AAL_NGR`<br />`queries.txt`<br />The `u_mul_matrix_sig%d/` folder |
| V-None            | The `sigma` value<br />The `eta` value<br />The `threads_num` value<br />The `dinx` value<br />The `time_budget`<br />`kkdesty_num_%d.json`<br />`M_edge_desty.json`<br />`vertices.txt`<br />`AAL_NGR`<br />`queries.txt`<br />The `u_mul_matrix_sig%d/` folder |

