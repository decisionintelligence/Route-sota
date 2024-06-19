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

4. You must also have what the program calls a "speedfile" or "the city map". The file included in the original test data is AAL_NGR. 

5. Running "get_tpath.py"

6. Running "get_overlap.py"

7. Running "get_vpath"

8. Running "get_vpath_desty.py"

9. Running "get_policy_U.py"

You now have All the files you need to run the routing algorithms. Before running one, remember to set all the paths and values to those that you desire. For an example:

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

