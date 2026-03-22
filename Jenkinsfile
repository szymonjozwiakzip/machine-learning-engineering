pipeline {
   agent any

   stages {
      stage('Checkout') {
         steps {
            checkout scm
         }
      }

      stage('Process dataset') {
         steps {
            sh 'bash ./process_data.sh'
         }
      }

      stage('Archive artifacts') {
         steps {
            archiveArtifacts artifacts: 'output_dataset/*.csv, process_log.txt', onlyIfSuccessful: true
         }
      }
   }
}