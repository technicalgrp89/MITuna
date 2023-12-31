import groovy.transform.Field

@Field String job_lim = "-A miopenConvolutionAlgoGEMM "

def rocmnode(name) {
    def node_name = 'tunatest'
    if(name == 'fiji') {
        node_name = 'tunatest && fiji';
    } else if(name == 'vega') {
        node_name = 'tunatest && vega';
    } else if(name == 'vega10') {
        node_name = 'tunatest && vega10';
    } else if(name == 'vega20') {
        node_name = 'tunatest && vega20';
    } else if(name == 'gfx908') {
        node_name = 'gfx908';
    } else {
        node_name = name
    }
    return node_name
}


def runsql(query) {
    echo "query: ${query}"
    def cmd = $/mysql --protocol tcp -h ${db_host} -u ${db_user} -p${db_password} "${db_name}" -e "${query}" -N -s /$
    def res = sh (script: "${cmd}", returnStdout: true).trim()
    return res
}

def buildSchema(){
    env.TUNA_DB_HOSTNAME = "${db_host}"
    env.TUNA_DB_NAME="${db_name}"
    env.TUNA_DB_USER_NAME="${db_user}"
    env.TUNA_DB_PASSWORD="${db_password}"
    env.gateway_ip = "${gateway_ip}"
    env.gateway_port = "${gateway_port}"
    env.gateway_user = "${gateway_user}"
    def cmd = $/mysql --protocol tcp -h ${db_host} -u ${db_user} -p${db_password} /$
    def drop_sql = $/  "DROP DATABASE IF EXISTS ${db_name};" /$
    def create_sql = $/ "CREATE DATABASE ${db_name};"/$
    sh "${cmd} -e ${drop_sql}"
    sh "${cmd} -e ${create_sql}"
    sh "./tuna/db_tables.py"
}

def cleanup() {
    def cmd = $/mysql --protocol tcp -h ${db_host} -u ${db_user} -p${db_password}  -e "DROP DATABASE IF EXISTS ${db_name}"/$
    sh "${cmd}"        
}

def getMachine() {
    def arch, cu, count
    for(String arch_cu :  sh(script:'bin/arch_cu.sh', returnStdout: true).split("\n")) { // is multiline
        (arch, cu, count) = arch_cu.tokenize('-')
        break
    }
    return [arch, cu]
}

def addMachine(arch, num_cu, machine_ip, machine_local_ip, username, pwd, port) {
    runsql("TRUNCATE machine;")
// TODO: this should come from different nodes
    runsql("INSERT INTO machine(hostname, local_ip, local_port,  avail_gpus, user, password, port, arch, num_cu, available, ipmi_inaccessible) VALUES(\'${machine_ip}\', \'${machine_local_ip}\', 22, \'0,1,2,3\', \'${username}\', \'${pwd}\', ${port}, \'${arch}\', ${num_cu}, TRUE, TRUE)" )
}


def addJobs() {
}

def finSolvers(){
    def tuna_docker = docker.build("ci-tuna:${branch_id}", "--build-arg FIN_TOKEN=${FIN_TOKEN} --build-arg BACKEND=HIPNOGPU .")
    /*
    Note: Does not need
            GPUs
    */
    tuna_docker.inside("--network host  --dns 8.8.8.8 ") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        buildSchema()

        sh "ls /opt/rocm/bin/fin"
        sh "ls /opt/rocm/bin/"
        sh "./tuna/go_fish.py --update_solvers"
        def num_solvers = runsql("SELECT count(*) from solver;")
        println "Number of solvers: ${num_solvers}"
        if (num_solvers.toInteger() == 0){
            error("Unable to add solvers from Fin")
        }
    }
}

def finApplicability(){
    def tuna_docker = docker.build("ci-tuna:${branch_id}", "--build-arg FIN_TOKEN=${FIN_TOKEN} --build-arg BACKEND=HIPNOGPU .")
    tuna_docker.inside("--network host  --dns 8.8.8.8") {
        checkout scm
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"

        sh "./tuna/go_fish.py --init_session -l new_session --arch gfx908 --num_cu 120"
        def sesh1 = 1 //runsql("select id from session order by id asc limit 1")
        sh "./tuna/go_fish.py --init_session -l new_session2 --arch gfx908 --num_cu 120"
        def sesh2 = 2 //runsql("select id from session order by id desc limit 1")

        sh "./tuna/import_configs.py -t recurrent_${branch_id} --mark_recurrent -f utils/recurrent_cfgs/alexnet_4jobs.txt"
        sh "./tuna/import_configs.py -t recurrent_${branch_id} --mark_recurrent -f utils/recurrent_cfgs/resnet50_4jobs.txt"

        sh "./tuna/import_configs.py -t recurrent_${branch_id}_nhwc --mark_recurrent -f utils/configs/conv_configs_NHWC.txt"
        sh "opentelemetry-instrument python3 ./tuna/import_configs.py -t recurrent_${branch_id}_nchw --mark_recurrent -f utils/configs/conv_configs_NCHW.txt"
        runsql("TRUNCATE table conv_solver_applicability")

        def num_cfg = runsql("SELECT count(*) from conv_config;")
        println "Count(*) conv_config table: ${num_cfg}"
        sh "opentelemetry-instrument python3 ./tuna/go_fish.py --update_applicability --session_id ${sesh1}"
        def num_solvers = runsql("SELECT count(*) from solver;")
        println "Number of solvers: ${num_solvers}"
        def num_sapp = runsql("SELECT count(*) from conv_solver_applicability where session=${sesh1};")
        println "Count(*) conv_solver_applicability table: ${num_sapp}"
        if (num_sapp.toInteger() == 0){
            error("Unable to get applicability from Fin for convolution")
        }

        sh "./tuna/import_configs.py -t recurrent_${branch_id}_bn --mark_recurrent -f utils/configs/batch_norm.txt -C batch_norm"
        runsql("TRUNCATE table bn_solver_applicability")
        def num_bn = runsql("SELECT count(*) from bn_config;")
        println "Count(*) bn_config table: ${num_bn}"

        sh "./tuna/go_fish.py --update_applicability --session_id ${sesh2} -C batch_norm"
        def num_sapp_bn = runsql("SELECT count(*) from bn_solver_applicability where session=${sesh2};")
        println "Count(*) bn_solver_applicability table: ${num_sapp_bn}"
        if (num_sapp_bn.toInteger() == 0){
            error("Unable to get applicability from Fin for batch norm")
        }
    }
}

def finFindCompile(){
    def tuna_docker = docker.build("ci-tuna:${branch_id}", "--build-arg FIN_TOKEN=${FIN_TOKEN} --build-arg BACKEND=HIPNOGPU .")
    tuna_docker.inside("--network host  --dns 8.8.8.8 ") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.PYTHONPATH=env.WORKSPACE
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        def sesh1 = runsql("select id from session order by id asc limit 1")

        sh "./tuna/import_configs.py -t recurrent_${branch_id} --mark_recurrent -f utils/recurrent_cfgs/alexnet_4jobs.txt"
        def num_cfg = runsql("SELECT count(*) from conv_config;")
        println "Count(*) conv_config table: ${num_cfg}"
        runsql("delete from conv_job;")
        runsql("alter table conv_job AUTO_INCREMENT=1;")
        sh "./tuna/load_job.py -l finFind_${branch_id} --all_configs --fin_steps \"miopen_find_compile, miopen_find_eval\" --session_id ${sesh1} ${job_lim}"
        def num_jobs = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}';").toInteger()
        sh "opentelemetry-instrument python3 ./tuna/go_fish.py --fin_steps miopen_find_compile -l finFind_${branch_id} --session_id ${sesh1}"
        def num_compiled_jobs = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}' AND state = 'compiled';").toInteger()
        sh "echo ${num_compiled_jobs} == ${num_jobs}"
        if (num_compiled_jobs != num_jobs){
            error("Unable to compile find jobs using Fin")
        }
        
        sh "./tuna/import_configs.py -t recurrent_${branch_id}_nhwc --mark_recurrent -f utils/configs/conv_configs_NHWC.txt"
        def num_cfg_nhwc = runsql("SELECT count(*) from conv_config;")
        println "Count(*) conv_config table: ${num_cfg_nhwc}"
        //runsql("delete from conv_job;")
        //runsql("alter table conv_job AUTO_INCREMENT=1;")
        sh "./tuna/load_job.py -l finFind_${branch_id}_nhwc -t recurrent_${branch_id}_nhwc --fin_steps \"miopen_find_compile, miopen_find_eval\" --session_id ${sesh1} ${job_lim}"
        def num_jobs_nhwc = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}_nhwc';").toInteger()
        sh "./tuna/go_fish.py --fin_steps miopen_find_compile -l finFind_${branch_id}_nhwc --session_id ${sesh1}"
        def num_compiled_jobs_nhwc = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}_nhwc' AND state = 'compiled';").toInteger()
        sh "echo ${num_compiled_jobs_nhwc} == ${num_jobs_nhwc}"
        if (num_compiled_jobs_nhwc != num_jobs_nhwc){
            error("Unable to compile find jobs using Fin")
        }
        sh "./tuna/import_configs.py -t recurrent_${branch_id}_nchw --mark_recurrent -f utils/configs/conv_configs_NCHW.txt"
        def num_cfg_nchw = runsql("SELECT count(*) from conv_config;")
        println "Count(*) conv_config table: ${num_cfg_nchw}"
        sh "./tuna/load_job.py -l finFind_${branch_id}_nchw -t recurrent_${branch_id}_nchw --fin_steps \"miopen_find_compile, miopen_find_eval\" --session_id ${sesh1} ${job_lim}"
        def num_jobs_nchw = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}_nchw';").toInteger()
        sh "./tuna/go_fish.py --fin_steps miopen_find_compile -l finFind_${branch_id}_nchw --session_id ${sesh1}"
        def num_compiled_jobs_nchw = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}_nchw' AND state = 'compiled';").toInteger()
        sh "echo ${num_compiled_jobs_nchw} == ${num_jobs_nchw}"
        if (num_compiled_jobs_nchw != num_jobs_nchw){
            error("Unable to compile find jobs using Fin")
        }
    }
}


def finFindEval(){
    def tuna_docker = docker.build("ci-tuna:${branch_id}", "--build-arg FIN_TOKEN=${FIN_TOKEN} --build-arg BACKEND=HIP .")
    tuna_docker.inside("--network host  --dns 8.8.8.8 ${docker_args}") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        def sesh1 = runsql("select id from session order by id asc limit 1")

        def num_jobs = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}' AND state = 'compiled';").toInteger()
        sh "opentelemetry-instrument python3 ./tuna/go_fish.py --fin_steps miopen_find_eval -l finFind_${branch_id} --session_id ${sesh1}"
        def num_evaluated_jobs = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}' AND state = 'evaluated';").toInteger()
        sh "echo ${num_evaluated_jobs} == ${num_jobs}"
        if (num_evaluated_jobs != num_jobs){
            error("Unable to evaluate find jobs using Fin")
        }
        def MIOPEN_BRANCH = runsql("SELECT miopen_v from session WHERE id=1;")
        def fdb_file = sh(script: "./tuna/export_db.py -a ${arch} -n ${num_cu} -f --session_id ${sesh1}", returnStdout: true)
        archiveArtifacts  "${fdb_file}"
        def kdb_file = sh(script: "./tuna/export_db.py -a ${arch} -n ${num_cu} -k --session_id ${sesh1}", returnStdout: true)
        archiveArtifacts "${kdb_file}"

        def num_jobs_nhwc = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}_nhwc' AND state = 'compiled';").toInteger()
        sh "./tuna/go_fish.py --fin_steps miopen_find_eval -l finFind_${branch_id}_nhwc --session_id ${sesh1}"
        def fdb_file_nhwc = sh(script: "./tuna/export_db.py -a ${arch} -n ${num_cu} -f --session_id ${sesh1} --filename fdb_nhwc", returnStdout: true)
        def num_evaluated_jobs_nhwc = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}_nhwc' AND state = 'evaluated';").toInteger()
        sh "echo ${num_evaluated_jobs_nhwc} == ${num_jobs_nhwc}"
        if (num_evaluated_jobs_nhwc != num_jobs_nhwc){
            error("Unable to evaluate find jobs using Fin")
        }

        archiveArtifacts  "${fdb_file_nhwc}"
        def kdb_file_nhwc = sh(script: "./tuna/export_db.py -a ${arch} -n ${num_cu} -k --session_id ${sesh1} --filename kdb_nhwc", returnStdout: true)
        archiveArtifacts "${kdb_file_nhwc}"

        def num_jobs_nchw = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}_nchw' AND state = 'compiled';").toInteger()
        sh "./tuna/go_fish.py --fin_steps miopen_find_eval -l finFind_${branch_id}_nchw --session_id ${sesh1}"
        def fdb_file_nchw = sh(script: "./tuna/export_db.py -a ${arch} -n ${num_cu} -f --session_id ${sesh1}", returnStdout: true)
        def num_evaluated_jobs_nchw = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}_nchw' AND state = 'evaluated';").toInteger()
        sh "echo ${num_evaluated_jobs_nchw} == ${num_jobs_nchw}"
        if (num_evaluated_jobs_nchw != num_jobs_nchw){
            error("Unable to evaluate find jobs using Fin")
        }

        archiveArtifacts  "${fdb_file_nchw}"
        def kdb_file_nchw = sh(script: "./tuna/export_db.py -a ${arch} -n ${num_cu} -k --session_id ${sesh1}", returnStdout: true)
        archiveArtifacts "${kdb_file_nchw}"
    }
}
def buildTunaDocker(){
    // The purpose of this job is to ensure that the Tuna Docker is uptodate on the eval/build machine for the CI jobs
    checkout scm
    def tuna_docker = docker.build("ci-tuna:${branch_id}")
    tuna_docker.inside("--network host "){
        sh "pwd"
    }
}

def loadJobTest() {
    def tuna_docker = docker.build("ci-tuna:${branch_id}", "--build-arg FIN_TOKEN=${FIN_TOKEN} .")
    tuna_docker.inside("--network host  --dns 8.8.8.8 ${docker_args}") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.PYTHONPATH=env.WORKSPACE
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        // setup version table
        runsql("SELECT * from machine;")
        echo "${arch} : ${num_cu}"
        def sesh1 = 1 //runsql("select id from session order by id asc limit 1")
        def sesh2 = 2 //runsql("select id from session order by id desc limit 1")

        sh "./tuna/import_configs.py -t recurrent_${branch_id} --mark_recurrent -f utils/recurrent_cfgs/alexnet_4jobs.txt"
        sh "./tuna/import_configs.py -t recurrent_${branch_id} --mark_recurrent -f utils/configs/conv_configs_NHWC.txt"
        def out = runsql("SELECT count(*) FROM conv_config_tags WHERE tag='recurrent_${branch_id}' ;")
        assert out.toInteger() > 0

        //reset job table
        runsql("DELETE FROM conv_job;")
        sh "./tuna/load_job.py -t recurrent_${branch_id} -l recurrent_${branch_id} --session_id ${sesh1} ${job_lim}"
        out = runsql("SELECT count(*) FROM conv_job WHERE reason='recurrent_${branch_id}' and session=${sesh1} ;")
        assert out.toInteger() > 0

        sh "./tuna/import_configs.py -t batch_norm_test -f utils/configs/batch_norm.txt -C batch_norm"
        // dump the added jobs for version 2
        def out_bn = runsql("SELECT count(*) FROM bn_config_tags WHERE tag='batch_norm_test' ;")
        assert out_bn.toInteger() > 0
        sh "./tuna/load_job.py -t batch_norm_test -l batch_norm_test -C batch_norm --session_id ${sesh2}"
        out_bn = runsql("SELECT count(*) FROM bn_job WHERE reason='batch_norm_test' and session=${sesh2} ;")
        assert out_bn.toInteger() > 0

        //reset jobs and test load solver
        runsql("DELETE FROM conv_job;")
        //runsql("INSERT INTO solver(solver, valid) VALUES ('ConvHipImplicitGemmV4R1Fwd', 1);")
        runsql("INSERT IGNORE INTO conv_solver_applicability(valid, applicable, config, solver, session) VALUES (1, 1, 1, 26, 1);")
        runsql("INSERT IGNORE INTO conv_solver_applicability(valid, applicable, config, solver, session) VALUES (1, 2, 1, 26, 1);")
        runsql("INSERT IGNORE INTO conv_solver_applicability(valid, applicable, config, solver, session) VALUES (1, 3, 1, 26, 1);")
        sh "./tuna/load_job.py -t recurrent_${branch_id} -l recurrent_${branch_id} -s ConvHipImplicitGemmV4R1Fwd --session_id ${sesh1}"
        out = runsql("SELECT count(*) FROM conv_job WHERE reason='recurrent_${branch_id}' and solver='ConvHipImplicitGemmV4R1Fwd' and session=${sesh1};")
        assert out.toInteger() > 0
    }
}

def perfCompile() {
    def tuna_docker = docker.build("ci-tuna:${branch_id}", "--build-arg FIN_TOKEN=${FIN_TOKEN} --build-arg BACKEND=HIPNOGPU .")
    tuna_docker.inside("--network host --dns 8.8.8.8 ${docker_args} ") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.TUNA_DOCKER_NAME="ci-tuna:${branch_id}"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        runsql("DELETE FROM conv_job;")
        def sesh1 = runsql("select id from session order by id asc limit 1")

        sh "./tuna/import_configs.py -t alexnet_${branch_id} --mark_recurrent -f utils/recurrent_cfgs/alexnet_4jobs.txt"
        sh "./tuna/load_job.py -t alexnet_${branch_id} -l alexnet_${branch_id} --session_id ${sesh1} --fin_steps miopen_perf_compile,miopen_perf_eval ${job_lim}"
        // Get the number of jobs
        def num_jobs = runsql("SELECT count(*) from conv_job where state = 'new' and reason = 'alexnet_${branch_id}'");
        sh "./tuna/go_fish.py --fin_steps miopen_perf_compile -l alexnet_${branch_id} --session_id ${sesh1}"
        def compiled_jobs = runsql("SELECT count(*) from conv_job where state = 'compiled' and reason = 'alexnet_${branch_id}';")
        if(compiled_jobs.toInteger() == 0)
        {
            error("Unable to compile any jobs for alexnet")
        }

        sh "./tuna/import_configs.py -t conv_${branch_id}_v2 --mark_recurrent -f utils/configs/conv_configs_NHWC.txt"
        sh "./tuna/import_configs.py -t conv_${branch_id}_v2 --mark_recurrent -f utils/configs/conv_configs_NCHW.txt"
        sh "./tuna/load_job.py -t conv_${branch_id}_v2 -l conv_${branch_id}_v2 --session_id ${sesh1} --fin_steps miopen_perf_compile,miopen_perf_eval ${job_lim}"
        // Get the number of jobs
        def num_conv_jobs = runsql("SELECT count(*) from conv_job where state = 'new' and reason = 'conv_${branch_id}_v2'");
        sh "./tuna/go_fish.py --fin_steps miopen_perf_compile -l conv_${branch_id}_v2 --session_id ${sesh1}"
        def compiled_conv_jobs = runsql("SELECT count(*) from conv_job where state = 'compiled' and reason = 'conv_${branch_id}_v2';")
        if(compiled_conv_jobs.toInteger() == 0)
        {
            error("Unable to compile any conv jobs")
        }
        echo "${compiled_conv_jobs}"
    }
}

def perfEval_gfx908() {
    def tuna_docker = docker.build("ci-tuna:${branch_id}", "--build-arg FIN_TOKEN=${FIN_TOKEN} .")
    tuna_docker.inside("--network host --dns 8.8.8.8 ${docker_args} ") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.TUNA_DOCKER_NAME="ci-tuna:${branch_id}"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        def sesh1 = runsql("select id from session order by id asc limit 1")

        def compiled_jobs = runsql("SELECT count(*) from conv_job where state = 'compiled' and reason = 'alexnet_${branch_id}';")
        sh "./tuna/go_fish.py --fin_steps miopen_perf_eval -l alexnet_${branch_id} --session_id ${sesh1}"
        def eval_jobs = runsql("SELECT count(*) from conv_job where state = 'evaluated' and reason = 'alexnet_${branch_id}';")
        if(eval_jobs.toInteger() != compiled_jobs.toInteger())
        {
            error("Unable to eval all jobs for alexnet")
        }

        def compiled_conv_jobs = runsql("SELECT count(*) from conv_job where reason = 'conv_${branch_id}_v2' and state = 'compiled';")
        sh "./tuna/go_fish.py --fin_steps miopen_perf_eval -l conv_${branch_id}_v2 --session_id ${sesh1}"
        def eval_conv_jobs = runsql("SELECT count(*) from conv_job where reason = 'conv_${branch_id}_v2' and state = 'evaluated';")
        def errored_conv_jobs = runsql("SELECT count(*) from conv_job where reason = 'conv_${branch_id}_v2' and state = 'errored';")
        if(eval_conv_jobs.toInteger() != compiled_conv_jobs.toInteger())
        {
            echo "#compiled jobs: ${compiled_conv_jobs}"
            echo "#evaluated jobs: ${eval_conv_jobs}"
            echo "#errored jobs: ${errored_conv_jobs}"
            error("Unable to eval all conv jobs")
        }

        def last_gold_v = runsql("SELECT max(golden_miopen_v) from conv_golden;")
        def next_gold_v = last_gold_v.toInteger() + 1
        sh "./tuna/update_golden.py --session_id ${sesh1} --golden_v ${next_gold_v} --base_golden_v ${last_gold_v}"
        def golden_entries = runsql("SELECT count(*) from conv_golden where session= ${sesh1};")
        def fdb_entries = runsql("SELECT count(*) from conv_golden where session= ${sesh1};")
        if(golden_entries.toInteger() != fdb_entries.toInteger())
        {
            echo "#fdb jobs: ${fdb_entries}"
            echo "#goden jobs: ${golden_entries}"
            error("FDB entries and golden entries do not match")
        }
    }
}

def pytestSuite1() {
    def tuna_docker = docker.build("ci-tuna:${branch_id}_pytest1", "--build-arg FIN_TOKEN=${FIN_TOKEN} .")
    tuna_docker.inside("--network host  --dns 8.8.8.8") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.TUNA_DOCKER_NAME="ci-tuna:${branch_id}_pytest1"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        addMachine(arch, num_cu, machine_ip, machine_local_ip, username, pwd, port)
        // download the latest perf db
        //runsql("DELETE FROM config_tags; DELETE FROM job; DELETE FROM config;")
        sshagent (credentials: ['bastion-ssh-key']) {                 
           sh "pytest tests/test_abort_file.py -s"
           sh "pytest tests/test_analyze_parse_db.py -s"

           sh "pytest tests/test_connection.py -s"
           // builder then evaluator in sequence
           sh "pytest tests/test_importconfigs.py -s"
           sh "pytest tests/test_machine.py -s"
           sh "pytest tests/test_dbBase.py -s"
           sh "pytest tests/test_driver.py -s"
           sh "pytest tests/test_fin_class.py -s"
           sh "pytest tests/test_fin_utils.py -s"
           sh "pytest tests/test_add_session.py -s"
           sh "pytest tests/test_merge_db.py -s"
           // The OBMC host used in the following test is down
           // sh "pytest tests/test_mmi.py "
        }
    }
}


def pytestSuite2() {
    def tuna_docker = docker.build("ci-tuna:${branch_id}_pytest2", "--build-arg FIN_TOKEN=${FIN_TOKEN} --build-arg BACKEND=HIPNOGPU .")
    tuna_docker.inside("--network host  --dns 8.8.8.8 ") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.TUNA_DOCKER_NAME="ci-tuna:${branch_id}_pytest2"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"

        addMachine(arch, num_cu, machine_ip, machine_local_ip, username, pwd, port)
        // download the latest perf db
        //runsql("DELETE FROM config_tags; DELETE FROM job; DELETE FROM config;")
        sshagent (credentials: ['bastion-ssh-key']) {                 
           // test fin builder and test fin builder conv in sequence
           sh "pytest tests/test_worker.py -s"
           sh "TUNA_LOGLEVEL=INFO pytest tests/test_fin_builder.py -s"
        }
    }
}

def pytestSuite3() {
    def tuna_docker = docker.build("ci-tuna:${branch_id}_pytest3", " --build-arg FIN_TOKEN=${FIN_TOKEN} --build-arg BACKEND=HIP .")
    tuna_docker.inside("--network host  --dns 8.8.8.8") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        //addMachine(arch, num_cu)
        // download the latest perf db
        //runsql("DELETE FROM config_tags; DELETE FROM job; DELETE FROM config;")
        sshagent (credentials: ['bastion-ssh-key']) {                 
           // test fin builder and test fin builder conv in sequence
           sh "pytest tests/test_fin_evaluator.py -s"
           sh "pytest tests/test_update_golden.py -s"
        }
    }
}

def runFormat() {
    node {
        checkout scm
        def tuna_docker = docker.build("ci-tuna:${branch_id}", "--build-arg FIN_TOKEN=${FIN_TOKEN} .")
        tuna_docker.inside("") {
            sh "yapf -d -r --style='{based_on_style: google, indent_width: 2}' tuna/ tests/ alembic/"
        }
    }
}

def runLint() {
    node {
          checkout scm
          def tuna_docker = docker.build("ci-tuna:${branch_id}", "--build-arg FIN_TOKEN=${FIN_TOKEN} .")
          tuna_docker.inside("") {
            sh "cd tuna && pylint -f parseable --max-args=8 --indent-string='  ' *.py"
          }
    }
}

def getJobReason()
{
  def job_reason = "${branch_name}_${miopen_branch_name}_${env.BUILD_ID}"
  return job_reason
}


def killContainer() {
  sh "srun --no-kill -p ${partition} -N 1-10 -l bash -c 'docker container list | grep  ${tuna_docker_name} | sed \"s#  #^#g\" | tr -s ^ | cut -d ^ -f 6 | xargs -I _ docker container kill _'"
  sh "srun --no-kill -p ${partition} -N 1-10 -l bash -c 'docker system prune -f'"
}

def getDockerName(backend)
{
  def tuna_docker_name = "${docker_registry}/ci-tuna:${branch_name}_${backend}_${env.BUILD_ID}"
  return tuna_docker_name
}

def LoadJobs()
{ 
  def script_args = ''
  def new_label = ''
  if(params.job_label == '')
  {
      new_label = getJobReason()
  }
  else
  {
      new_label = params.job_label
  }
  script_args = script_args + ' -l ' + "${new_label}"
  if(params.cmd != '')
  {
      script_args = script_args + " --cmd ${params.cmd} "
  }
  if(params.stage == 'fin_find')
  {
      script_args = script_args + " --fin_steps \"miopen_find_compile, miopen_find_eval\""
  }
  else if(params.stage == 'perf')
  {
      script_args = script_args + " --fin_steps \"miopen_perf_compile, miopen_perf_eval\""
  }
  if(params.all_configs)
  {
      script_args = script_args + " --all_configs "
  }
  else
  {
      script_args = script_args + " -t ${params.config_tag} "
  }
  echo "${script_args}"

  def build_args = "--build-arg FIN_TOKEN=${FIN_TOKEN} --build-arg ROCMVERSION=${params.rocm_version} --build-arg OSDB_BKC_VERSION=${params.osdb_bkc_version} --build-arg BACKEND=HIPNOGPU --build-arg MIOPEN_BRANCH=${miopen_branch_name} --build-arg DB_NAME=${params.db_name} --build-arg DB_USER_NAME=${db_user} --build-arg DB_USER_PASSWORD=${db_password} --build-arg DB_HOSTNAME=${db_host} ."
  if(params.base_image != '')
  {
    build_args = build_args + " --build-arg BASEIMAGE=${params.base_image} --build-arg ROCM_PRE=1"
  }
  def docker_run_args = "--network host --dns 8.8.8.8 -e TUNA_DB_HOSTNAME=${db_host} -e TUNA_DB_NAME=${params.db_name} -e TUNA_DB_USER_NAME=${db_user} -e TUNA_DB_PASSWORD=${db_password} -e gateway_ip=${gateway_ip} -e gateway_port=${gateway_port} -e gateway_user=${gateway_user} -e TUNA_LOGLEVEL=${params.tuna_loglevel}"

  sh "echo ${build_args}"
  def tuna_docker = docker.build("${tuna_docker_name}", "${build_args} ." )
  tuna_docker.inside("${docker_run_args}") {
      env.PYTHONPATH=env.WORKSPACE
      env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
      env.TUNA_LOGLEVEL="${tuna_loglevel}" 

      echo "/tuna/tuna/load_job.py --session_id ${params.session_id} ${script_args}"
      sh "python3 /tuna/tuna/load_job.py --session_id ${params.session_id} ${script_args}"
  }
  tuna_docker.push()
}


def getSessionVals(session_id)
{
  String res = runsql("select arch, num_cu, rocm_v, miopen_v from session where id=${session_id};")

  def arch = res.split("[ \t]+")[0]
  def num_cu = res.split("[ \t]+")[1]
  def rocm_v = res.split("[ \t]+")[2]
  def miopen_v = res.split("[ \t]+")[3]
  echo "$arch $num_cu $rocm_v $miopen_v"

  def partition = "${arch}_${num_cu}"

  def osdb_bkc_version = ''
  def rocm_version = ''
  def subv_i = rocm_v.indexOf('-')
  if(subv_i >= 0)
  {
    osdb_bkc_version=rocm_v.substring(subv_i+1)
  }
  else
  {
    rocm_version = rocm_v
  }

  subv_i = miopen_v.indexOf('-dirty')
  if(subv_i >= 0)
  {
    miopen_v = miopen_v.substring(0, subv_i)
  }

  return [partition, osdb_bkc_version, rocm_version, miopen_v]
}


def applicUpdate(){
  def tuna_docker
  (_, osdb_bkc_version, rocm_version, miopen_v) = getSessionVals(params.session_id)

  def build_args = "--build-arg FIN_TOKEN=${FIN_TOKEN} --network host --build-arg ROCMVERSION=${rocm_version} --build-arg OSDB_BKC_VERSION=${osdb_bkc_version} --build-arg BACKEND=${backend} --build-arg MIOPEN_BRANCH=${miopen_v} --build-arg DB_NAME=${params.db_name} --build-arg DB_USER_NAME=${params.db_user} --build-arg DB_USER_PASSWORD=${params.db_password} --build-arg DB_HOSTNAME=${params.db_host} --build-arg MIOPEN_USE_MLIR=${params.use_mlir}"

  if(params.base_image != '')
  {
    build_args = build_args + " --build-arg BASEIMAGE=${params.base_image} --build-arg ROCM_PRE=1"
  }
  sh "echo ${build_args}"

  tuna_docker = docker.build("${tuna_docker_name}", "${build_args} ." )
  tuna_docker.push()

  def docker_args = "--network host  --dns 8.8.8.8 -e TUNA_DB_HOSTNAME=${db_host} -e TUNA_DB_NAME=${params.db_name} -e TUNA_DB_USER_NAME=${db_user} -e TUNA_DB_PASSWORD=${db_password} -e gateway_ip=${gateway_ip} -e gateway_port=${gateway_port} -e gateway_user=${gateway_user} -e TUNA_LOGLEVEL=${params.tuna_loglevel}"

  def use_tag = ''
  if(params.config_tag != '')
  {
    use_tag = "-l '${params.config_tag}'"
  }

  if(params.UPDATE_SOLVERS)
  {
    sh "srun --no-kill -p build-only -N 1 -l bash -c 'docker run ${docker_args} ${tuna_docker_name} ./tuna/go_fish.py --update_solvers'"
    def num_solvers = runsql("SELECT count(*) from solver;")
    println "Number of solvers: ${num_solvers}"
    if (num_solvers.toInteger() == 0){
        error("Unable to add solvers from Fin")
    }
  }
  if(params.UPDATE_APPLICABILITY)
  {
    sh "srun --no-kill -p build-only -N 1 -l bash -c 'docker run ${docker_args} ${tuna_docker_name} ./tuna/go_fish.py --update_applicability --session_id ${params.session_id} ${use_tag}'"
    def num_sapp = runsql("SELECT count(*) from conv_solver_applicability where session=${params.session_id};")
    println "Session ${params.session_id} applicability: ${num_sapp}"
    if (num_sapp.toInteger() == 0){
      error("Unable to get applicability from Fin")
    }
  }

}


def compile()
{
  def tuna_docker
  (_, osdb_bkc_version, rocm_version, miopen_v) = getSessionVals(params.session_id)

  def build_args = " --network host --build-arg FIN_TOKEN=${FIN_TOKEN} --build-arg ROCMVERSION=${rocm_version} --build-arg OSDB_BKC_VERSION=${osdb_bkc_version} --build-arg BACKEND=${backend} --build-arg MIOPEN_BRANCH=${miopen_v} --build-arg DB_NAME=${params.db_name} --build-arg DB_USER_NAME=${params.db_user} --build-arg DB_USER_PASSWORD=${params.db_password} --build-arg DB_HOSTNAME=${params.db_host} --build-arg MIOPEN_USE_MLIR=${params.use_mlir}"

  if(params.base_image != '')
  {
    build_args = build_args + " --build-arg BASEIMAGE=${params.base_image} --build-arg ROCM_PRE=1"
  }

  sh "echo ${build_args}"
  tuna_docker = docker.build("${tuna_docker_name}", "${build_args} ." )

  tuna_docker.inside("--network host  --dns 8.8.8.8 ") {
      env.PYTHONPATH=env.WORKSPACE
      env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
      env.TUNA_LOGLEVEL="${tuna_loglevel}"
      sh "pwd"
  }
  // push the image 
  tuna_docker.push()
  env_list = params.env.split(' ')
  for(item in env_list)
  {
    if(item.replaceAll("\\s","") != "")
    {
      if(item.contains("="))
      {
        docker_args += " -e ${item}"
      }
      else
      {
        error "Not added to env: ${item}"
      }
    }
  }

  if(params.stage == 'perf')
  {
    docker_args += " --privileged --device=/dev/kfd --device /dev/dri:/dev/dri:rw --volume /dev/dri:/dev/dri:rw --group-add video"
  }

  def compile_cmd = ''
  if(params.stage == 'fin_find')
  {
    compile_cmd = '--fin_steps miopen_find_compile'
  }
  else
  {
    compile_cmd = '--fin_steps miopen_perf_compile'
  }
  if(params.dynamic_solvers_only)
  {
    compile_cmd += ' --dynamic_solvers_only'
  }

  // Run the jobs on the cluster
  sh "srun --no-kill -p ${partition} -N 1-10 -l bash -c 'docker run ${docker_args} ${tuna_docker_name} python3 /tuna/tuna/go_fish.py ${compile_cmd} --session_id ${params.session_id}'"
}


def evaluate(params)
{
  def tuna_docker
  (partition, osdb_bkc_version, rocm_version, miopen_v) = getSessionVals(params.session_id)

  def build_args = " --network host --build-arg FIN_TOKEN=${FIN_TOKEN} --build-arg ROCMVERSION=${rocm_version} --build-arg OSDB_BKC_VERSION=${osdb_bkc_version} --build-arg BACKEND=HIP --build-arg MIOPEN_BRANCH=${miopen_v} --build-arg DB_NAME=${params.db_name} --build-arg DB_USER_NAME=${params.db_user} --build-arg DB_USER_PASSWORD=${params.db_password} --build-arg DB_HOSTNAME=${params.db_host} --build-arg MIOPEN_USE_MLIR=${params.use_mlir}"

  if(params.base_image != '')
  {
    build_args = build_args + " --build-arg BASEIMAGE=${params.base_image} --build-arg ROCM_PRE=1"
  }

  sh "echo ${build_args}"
  tuna_docker = docker.build("${tuna_docker_name}", "${build_args} ." )
  tuna_docker.push()

  env_list = params.env.split(' ')
  for(item in env_list)
  {
    if(item.replaceAll("\\s","") != "")
    {
      if(item.contains("="))
      {
        docker_args += " -e ${item}"
      }
      else
      {
        error "Not added to env: ${item}"
      }
    }
  }
  def eval_cmd = ''
  if(params.stage == 'fin_find')
  {
    eval_cmd = '--fin_steps miopen_find_eval'
  }
  else
  {
    eval_cmd = '--fin_steps miopen_perf_eval'
  }
  if(params.dynamic_solvers_only)
  {
    eval_cmd += ' --dynamic_solvers_only'
  }

  sh "srun --no-kill -p ${partition} -N 1-10 -l bash -c 'docker run ${docker_args} ${tuna_docker_name} python3 /tuna/tuna/go_fish.py ${eval_cmd} --session_id ${params.session_id} || scontrol requeue \$SLURM_JOB_ID'"
}

def doxygen() {
    node {
          checkout scm
          def tuna_docker = docker.build("ci-tuna:${branch_id}", "--build-arg FIN_TOKEN=${FIN_TOKEN} .")
          tuna_docker.inside("") {
            sh "cd doc && doxygen Doxyfile"
            def empty = sh returnStdout: true, script: "ls doc | wc -l"
            echo "${empty}"
            if (empty.toInteger() == 0){
              error("Unable to generate Doxygen file")
            }
          }
    }
}





