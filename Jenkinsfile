pipeline {
    agent any

    parameters {
        string(
            name: 'CUTOFF', 
            defaultValue: '500', 
            description: 'Liczba wierszy do zachowania ze zbioru danych',
            trim: true
        )
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Create dataset in Dockerfile container') {
            agent {
                dockerfile {
                    filename 'Dockerfile'
                    dir '.'
                    reuseNode true
                }
            }
            steps {
                sh 'python3 ./lab01.py create-dataset --cutoff ${CUTOFF}'
            }
        }

        stage('Archive artifacts') {
            steps {
                archiveArtifacts artifacts: 'output_dataset/*.csv, prepared_data/*.csv, process_log.txt', allowEmptyArchive: true
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'process_log.txt', allowEmptyArchive: true
        }
    }
}