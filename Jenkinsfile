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

        stage('Process dataset') {
            steps {
                sh "bash ./process_data.sh ${params.CUTOFF}"
            }
        }

        stage('Archive artifacts') {
            steps {
                archiveArtifacts artifacts: 'output_dataset/*.csv, process_log.txt', onlyIfSuccessful: true
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'process_log.txt', allowEmptyArchive: true
        }
    }
}