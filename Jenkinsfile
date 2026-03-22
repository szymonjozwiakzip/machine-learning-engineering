pipeline {
   agent any
   //Definijuemy parametry, które będzie można podać podczas wywoływania zadania
   parameters {
     string (
         defaultValue: 'Hello World!',
         description: 'Tekst, którym chcesz przywitać świat',
         name: 'INPUT_TEXT',
         trim: false
        )
   }
   stages {
      stage('Hello') {
         steps {
            //Wypisz wartość parametru w konsoli (To nie jest polecenie bash, tylko groovy!)
            echo "INPUT_TEXT: ${INPUT_TEXT}"
            //Wywołaj w konsoli komendę "figlet", która generuje ASCII-art
            sh "echo \"${INPUT_TEXT}\" > output.txt"
         }
      }
      stage('Goodbye!') {
         steps {
            echo 'Goodbye!'
            //Zarchiwizuj wynik
            archiveArtifacts 'output.txt'
         }
      }
   }
}