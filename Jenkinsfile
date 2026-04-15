pipeline {
    agent any

    parameters {
        string(
            name: 'CUTOFF',
            defaultValue: '500',
            description: 'Liczba wierszy do zachowania ze zbioru danych (0 = wszystkie)',
            trim: true
        )
        string(
            name: 'EPOCHS',
            defaultValue: '80',
            description: 'Maksymalna liczba epok trenowania',
            trim: true
        )
        string(
            name: 'HIDDEN_LAYERS',
            defaultValue: '64,32',
            description: 'Rozmiary warstw ukrytych sieci (oddzielone przecinkami, np. 128,64,32)',
            trim: true
        )
        choice(
            name: 'ACTIVATION',
            choices: ['relu', 'tanh', 'logistic'],
            description: 'Funkcja aktywacji neuronów'
        )
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Create dataset') {
            agent {
                dockerfile {
                    filename 'Dockerfile'
                    dir '.'
                    reuseNode true
                }
            }
            steps {
                sh "python3 ./lab01.py create-dataset --cutoff ${params.CUTOFF}"
            }
        }

        stage('Train model') {
            agent {
                dockerfile {
                    filename 'Dockerfile'
                    dir '.'
                    reuseNode true
                }
            }
            steps {
                sh """
                    python3 ./train_nn.py \
                        --epochs ${params.EPOCHS} \
                        --hidden-layers '${params.HIDDEN_LAYERS}' \
                        --activation ${params.ACTIVATION} \
                        --model models/hotel_mlp.joblib
                """
            }
        }

        stage('Archive model') {
            steps {
                archiveArtifacts artifacts: 'models/*.joblib', allowEmptyArchive: false
            }
        }
    }
}
