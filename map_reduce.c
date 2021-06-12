#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/time.h>
#include <limits.h>
#include <assert.h>
#include <string.h>

//#define NUMPAT 4
#define NUMIN  2
#define NUMHID 3
#define NUMOUT 1
#define NUM_THREADS 4
#define EPOCHS 500

typedef struct paire
{
	int key;
	double value;
} paire;

typedef struct groupe
{
    double *tab_val;
    double partial_g;
    int key;

} groupe;




typedef struct timezone timezone_t;
typedef struct timeval timeval_t;

timeval_t t1, t2;
timezone_t tz;


static struct timeval _t1, _t2;
static struct timezone _tz;
timeval_t t1, t2;
timezone_t tz;

static unsigned long _temps_residuel = 0;
#define top1() gettimeofday(&_t1, &_tz)
#define top2() gettimeofday(&_t2, &_tz)

void init_cpu_time(void)
{
   top1(); top2();
   _temps_residuel = 1000000L * _t2.tv_sec + _t2.tv_usec -
                     (1000000L * _t1.tv_sec + _t1.tv_usec );
}

unsigned long cpu_time(void) 
{
   return 1000000L * _t2.tv_sec + _t2.tv_usec -
           (1000000L * _t1.tv_sec + _t1.tv_usec ) - _temps_residuel;
}

pthread_mutex_t mutex_sum; //creation du mutex
paire *tab; // pour stoker les resultats d'un mappers
paire **map_tab;// contient le resultat de tous les mappers; chaque element etant un tab
int nb_groupes = 0;
groupe *tab_groupe;


int i, j, k, p, np, op, epoch; 
int *ranpat ; // ranpat est un tableau
    
int  NumPattern , NumInput = NUMIN, NumHidden = NUMHID, NumOutput = NUMOUT;
double **Input, **Target, **SumH , **Hidden, **SumO, **Output;
double WeightIH[NUMIN+1][NUMHID+1],WeightHO[NUMHID+1][NUMOUT+1];
double DeltaO[NUMOUT+1], SumDOW[NUMHID+1], DeltaH[NUMHID+1];
double DeltaWeightIH[NUMIN+1][NUMHID+1], DeltaWeightHO[NUMHID+1][NUMOUT+1];
double Error, eta = 0.5, alpha = 0.9, smallwt = 0.5;

#define rando() ((double)rand()/((double)RAND_MAX+1))

pthread_mutex_t mutex_sum;// variable de gestion de la session critique



int  nombrelignes(char* file_name)
{
	FILE *fp=fopen(file_name,"r");

	assert(fp!=NULL);//ouverture du fichier ok
	char* line=NULL;
	size_t size=0;// taille de la ligne
	int elements =0;

	while(getline(&line,&size,fp)!=-1)
	{	
		if (line[0] != '\n')
		{
			elements++;	
		}
	}
	free(line);
	fclose(fp);
	return elements;
}

void suprRetourLigne(char *str) // eliminer le retour a la ligne que getline ajoute dans la chaine de caractere.
{
	int i = 0;

   while (str[i] != '\0')
   {
      if ( str[i] == '\n') 
      {
         str[i] = '\0';
      }

      i++;
   }

}


void Lecturefichier(char* file_name)
{
	FILE *fp=fopen(file_name,"r");
    char *mot, *mot2;
	int compteur = 0, compteur2 = 0; 
    int flag =0;
	assert(fp!=NULL);
	char* line=NULL;
	size_t size=0;
	int nb_elements = nombrelignes(file_name);

	while(getline(&line,&size,fp)!=-1)
	{
		if (line[0] != '\n')
		{
			suprRetourLigne(line);
            
			

            while((mot=strsep(&line,","))!=NULL)// on separe les donnees en inputs , target
            {
                if (flag==0)// on recupere les inputs
                {
                    
                     while((mot2=strsep(&mot,";"))!=NULL)
                     {
                         Input[compteur][compteur2]=atof(mot2); // atof converti une chaine de caractere en double;
                         compteur2++;
                     }
                    
                    flag++;
                }
                else  // on recupere les targets
                { 
                    while((mot2=strsep(&mot,";"))!=NULL)
                     {
                         Target[compteur][compteur2]=atof(mot2); // atof converti une chaine de caractere en double;
                         compteur2++;
                     }
                    
                    
                }
                compteur2 =0;

            }

	
			compteur++;
            flag =0;
		}
	}
	free(line);
	fclose(fp);
}



void *map(void *arg) // il prend en parametre l'id du thread
{

    int i, taille;
    long debut, fin;
    long indice;
    int cle;

    double SumH[NumPattern + 1][NumHidden + 1], Hidden[NumPattern + 1][NumHidden + 1];
    double SumO[NumPattern + 1][NumOutput + 1], Output[NumPattern + 1][NumOutput + 1];
    double DeltaO[NumOutput + 1], SumDOW[NumHidden + 1], DeltaH[NumHidden + 1];

    //indice ou ID du thread
    indice = (long)arg;

    //Indice de début dans la table des patterns à considérer
    debut = indice * (NumPattern / NUM_THREADS);

    // printf("\n================ Thread d'ID %ld ====================\n", indice);

    //indice de fin dans la table des patterns
    if (indice == (NUM_THREADS - 1))
        fin = NumPattern;
    else
        fin = debut + (NumPattern / NUM_THREADS);

    pthread_mutex_lock(&mutex_sum); //Activation du mutex pour evite les confusion dans les données
    Error = 0.0;

    /*initialisation des DeltaO et deltaH*/
    for (k = 0; k <= NumOutput; k++)
    {

        DeltaO[k] = 0.0;
    }

    for (j = 0; j <= NumHidden; j++)
    {

        DeltaH[j] = 0.0;
    }

    for (np = debut + 1; np <= fin; np++)
    {
        // printf("\n++++Exemple Numero %d+++++\n", np);
        /* repeat for all the training patterns */
        p = ranpat[np];

        // printf("\nExemple Numero %d\n", p);

        // printf("\n\nActivations couche cachée\n");
        for (j = 1; j <= NumHidden; j++)
        {
            SumH[p][j] = WeightIH[0][j];
            for (i = 1; i <= NumInput; i++)
            {
                // printf("\nInput[%d][%d] = %f\n", p,i, Input[p][i]);
                SumH[p][j] += Input[p][i] * WeightIH[i][j];
            }
            Hidden[p][j] = 1.0 / (1.0 + exp(-SumH[p][j]));
            // printf("\nHidden[%d][%d] = %f\n", p,j, Hidden[p][j]);
        }

        // printf("\n\nActivations couche sortie\n");
        for (k = 1; k <= NumOutput; k++)
        {
            SumO[p][k] = WeightHO[0][k];
            for (j = 1; j <= NumHidden; j++)
            {
                SumO[p][k] += Hidden[p][j] * WeightHO[j][k];
            }
            Output[p][k] = 1.0 / (1.0 + exp(-SumO[p][k]));

            // printf("\nOutput[%d][%d] = %f\n", p,k, Output[p][k]);

            Error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]); /* SSE */

            // printf("\n\n========= Output et Target ==========\n\n");
            // printf("\nTarget[%d][%d] == %f\n", p,k, Target[p][k]);

            // printf("\nDeltaO temporaire %d = %f\n", k, (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]));

            DeltaO[k] = DeltaO[k] + ((Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]));

            // printf("\nDeltaO[%d] = %f\n", k, DeltaO[k]);
        }

        /* 'back-propagate' errors to hidden layer*/
        for (j = 1; j <= NumHidden; j++)
        {
            SumDOW[j] = 0.0;
            for (k = 1; k <= NumOutput; k++)
            {
                SumDOW[j] += WeightHO[j][k] * DeltaO[k];
            }
            // printf("\nDeltaH temporaire %d = %f\n", j, SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]));
            DeltaH[j] = DeltaH[j] + (SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]));
            // printf("\nDeltaH[%d] = %f\n", j, DeltaH[j]);
        }
    }

    // for (i = 1; i <= NumHidden; i++)
    // {
    //     printf("\nDeltaH %d = %f\n", i, DeltaH[i]);
    // }
    // for (i = 1; i <= NumOutput; i++)
    // {
    //     printf("\nDelta0 %d = %f\n", i, DeltaO[i]);
    // }

    // stockage des resultats des mappers

    // printf("\n\n======= Stockage dans le Map\n==========");
    for (j = 1; j <= NumHidden; j++)
    {
        map_tab[indice][j].key = j;
        map_tab[indice][j].value = DeltaH[j];
        // printf("\ntable[%d].value = %f\n", j, table[j].value);
    }
    for (k = 1; k <= NumOutput; k++)
    {
        cle = k + NUMIN;
        map_tab[indice][cle].key = cle;
        map_tab[indice][cle].value = DeltaO[k];

        // printf("\ntable[%d].value = %f\n", cle, table[cle].value);
    }

    // map_tab[indice] = table; // on stocke le resultat de chaque map;

    pthread_mutex_unlock(&mutex_sum);

    pthread_exit((void *)0);
}

int verifierpresence(int key ,int taille)
{

    int i;
               

      if (taille == 0)
      {

        return 0;

     } 
     else
    {
        for(i=0; i<taille;i++) 
        {
                                
                if (key == tab_groupe[i].key)
                {
                    return 1;
                    printf("ok\n");
                }

        }    

    }

    return 0;
            
            
}


double* realloc_s (double **ptr, size_t taille) // cette fonction permet de realouer(redefinir) la taille d'un tableau, faire les tableau dynamique
// cette fonction garde les donees
{
    double *ptr_realloc = realloc(*ptr, taille);

    if (ptr_realloc != NULL)
        *ptr = ptr_realloc;


        return ptr_realloc;
}


double *tab_val_groupe(int key)
{
    double *tab_elements = NULL;
    int nombre_element = 0;
    int i = 0,j=0,k=0,cle;

    for (i = 0; i < NUM_THREADS; ++i)
    {
                for( j = 1 ; j <= NumHidden ; j++ )
                {
                    if (map_tab[i][j].key  == key)
                    {
                        nombre_element++;
                        realloc_s(&tab_elements,nombre_element*sizeof(double));
//                        tab_elements[nombre_element-1] = map_tab[i][j].value;
                        tab_elements[nombre_element-1] = map_tab[i][j].value;
                                             
                    }
               
                }

                for( k = 1 ; k <= NumOutput ; k++ ) 
                {   
                    cle = k + NUMIN;

                    if (map_tab[i][cle].key  == key)
                    {
                        nombre_element++;
                        realloc_s(&tab_elements,nombre_element*sizeof(double));
//                        tab_elements[nombre_element-1] = map_tab[i][cle].value;
                        tab_elements[nombre_element-1] = map_tab[i][cle].value;
                        
                    }                

                  
                }
            

    }

    nombre_element++;
    realloc_s(&tab_elements,nombre_element*sizeof(double));
//    tab_elements[nombre_element-1] = -1.0;
     tab_elements[nombre_element-1] = -1.0;

    return tab_elements;
}



void afficherListeGroupe() // affiche la liste des groupes
{
    int i = 0;
    int j = 0;
        

        for(i=0; i<nb_groupes;i++) 
        {
            j = 0;


            printf("(%d,[", tab_groupe[i].key);

            while(tab_groupe[i].tab_val[j+1] != -1.0)
            {
                printf("%f,",tab_groupe[i].tab_val[j]);
                j++;

            }
            printf("%f",tab_groupe[i].tab_val[j]);
            printf("])->");

        }

        printf("fin\n");
      
}


void regroupement()
{

    int i = 0, k = 0, j = 0, cle, compteur = 0;

    // tab_groupe[compteur].partial_g = 0.0;

    for (i = 1; i <= NumHidden + NumOutput; i++)
    {
        tab_groupe[i].partial_g = 0.0;
    }

    // for(i=0; i<NUM_THREADS; i++) {
    //     tab = map_tab[i];
    //     printf("\nValeur Map %d\n", i);
    //     for(j = 1; j<=NumHidden+NumOutput; j++) {
    //         printf("\n(%d, %f)\n", tab[j].key, tab[j].value);
    //     }
    // }
    // printf("\nMap_tab[0][3].value = %f\n", map_tab[0][3].value);
    // printf("\nMap_tab[1][3].value = %f\n", map_tab[1][3].value);
    // printf("\nMap_tab[2][3].value = %f\n", map_tab[2][3].value);

    // printf("\nMap_tab[0][1].value = %f\n", map_tab[0][1].value);
    // printf("\nMap_tab[1][1].value = %f\n", map_tab[1][1].value);
    // printf("\nMap_tab[2][1].value = %f\n", map_tab[2][1].value);

    // printf("\nMap_tab[0][2].value = %f\n", map_tab[0][2].value);
    // printf("\nMap_tab[1][2].value = %f\n", map_tab[1][2].value);
    // printf("\nMap_tab[2][2].value = %f\n", map_tab[2][2].value);

    for (i = 0; i < NUM_THREADS; i++)
    {
        compteur = 1;
        tab = map_tab[i];
        // printf("\n\nPassage %d map\n", i);

        for (j = 1; j <= NumHidden; j++)
        {

            tab_groupe[compteur].key = tab[j].key;
            tab_groupe[compteur].partial_g = tab_groupe[compteur].partial_g + tab[j].value;

            // printf("\ntab_groupe[%d].partial_g = %f\n", compteur,tab_groupe[compteur].partial_g );
            compteur++;
            nb_groupes = compteur;
        }

        for (k = 1; k <= NumOutput; k++)
        {

            cle = k + NUMIN;
            // printf("\ncompteur = %d\n", compteur);
            tab_groupe[compteur].key = tab[cle].key;

            // printf("\ntab[%d].value = %f\n", cle,tab[cle].value );
            tab_groupe[compteur].partial_g = tab_groupe[compteur].partial_g + tab[cle].value;

            // printf("\ntab_groupe[%d].partial_g = %f\n", compteur,tab_groupe[compteur].partial_g );
            compteur++;
            nb_groupes = compteur;
        }
    }

    // printf("\n=========Fin du regroupement=========\n");
    // for(i=1; i<=NumHidden+NumOutput; i++) {
    //     printf("\ntab_groupe[%d].partial_g = %f\n", i,tab_groupe[i].partial_g);
    // }
}



void reduce()
{
    int i, j, k, t, compteur, cle;
    double somme = 0.0;
    double tmp = 0.0;

    for (j = 1; j <= NumHidden; j++)
    {
        // printf("\nContenu de tab_groupe[%d].partial_g = %f \n",j,tab_groupe[j].partial_g );
        WeightIH[0][j] = WeightIH[0][j] - eta * tab_groupe[j].partial_g;
        for (i = 1; i <= NumInput; i++)
        {
            WeightIH[i][j] = WeightIH[i][j] - eta * tab_groupe[j].partial_g;
        }
    }

    for (k = 1; k <= NumOutput; k++)
    {
        cle = k + NumHidden;

        // printf("\nContenu de tab_groupe[%d].partial_g = %f \n", cle,tab_groupe[cle].partial_g );

        WeightHO[0][k] = WeightHO[0][k] - eta * tab_groupe[cle].partial_g;
        for (j = 1; j <= NumHidden; j++)
        {
            // printf("\n Valeur de p = %d\n", p);

            // printf("\neta = %f \n tab_groupe[%d].partial_g = %f\n(1/%d) = %f\nResultat = %f\n", eta, cle,tab_groupe[cle].partial_g, NumPattern, (1.0 / NumPattern), eta * tab_groupe[cle].partial_g * (1.0 / NumPattern));

            WeightHO[j][k] = WeightHO[j][k] - eta * tab_groupe[cle].partial_g;

            // printf("\nPoids %d %d = %f\n", j, k, WeightHO[j][k]);
        }
    }

    // printf("\n========== Fin Reduce ===========\n");
    // printf("\nPoids couche entrée et couche cachée\n");
    // for (i = 1; i <= NumHidden; i++)
    // {
    //     for (j = 0; j <= NumInput; j++)
    //     {

    //         printf("\nPoids %d %d = %f\n", j, i, WeightIH[i][j]);
    //     }
    // }
    // printf("\nPoids couche cachée et couche sortie\n");

    // for (k = 1; k <= NumOutput; k++)
    // { /* initialize WeightHO and DeltaWeightHO */
    //     for (j = 0; j <= NumHidden; j++)
    //     {

    //         printf("\nPoids %d %d = %f\n", k, j, WeightHO[j][k]);
    //     }
    // }
}
          





void sequential()
{ 

    /* initialize WeightIH and DeltaWeightIH */
    for( j = 1 ; j <= NumHidden ; j++ ) 
    {   
        for( i = 0 ; i <= NumInput ; i++ ) { 
            DeltaWeightIH[i][j] = 0.0 ;
            WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
    
    
    
    /* initialize WeightHO and DeltaWeightHO */
    for( k = 1 ; k <= NumOutput ; k ++ ) 
    {    
        for( j = 0 ; j <= NumHidden ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;              
            WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }

    epoch=0;
    Error=1.0;
     
    while(epoch<EPOCHS )
    {    

            /* iterate weight updates */
            for( p = 1 ; p <= NumPattern ; p++ ) 
            {   
                /* randomize order of individuals */
                ranpat[p] = p ;
            }
            for( p = 1 ; p <= NumPattern ; p++) 
            {
                np = p + rando() * ( NumPattern + 1 - p ) ;
                op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
            }

            Error = 0.0 ;

            /* repeat for all the training patterns */
            for( np = 1 ; np <= NumPattern ; np++ ) 
            {   


                p = ranpat[np];
                
                
                /*initialisation des DeltaO et deltaH*/  
                for( k = 0 ; k <= NumOutput ; k++ ) { 
                    
                    DeltaO[k] =0.0;
                }
                
                 
                for( j = 0 ; j <= NumHidden ; j++ ) {   
                   
                    DeltaH[j] =0.0 ;
                }
                
            
                /* compute hidden unit activations, feed-forward 1 */
                for( j = 1 ; j <= NumHidden ; j++ ) {    
                    SumH[p][j] = WeightIH[0][j] ;
                    for( i = 1 ; i <= NumInput ; i++ ) {
                        SumH[p][j] += Input[p][i] * WeightIH[i][j] ;
                    }
                    Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
                }
                
                
                /* compute output unit activations and errors feed_forward 2 */
                for( k = 1 ; k <= NumOutput ; k++ ) 
                { 
                    SumO[p][k] = WeightHO[0][k] ;
                    for( j = 1 ; j <= NumHidden ; j++ ) {
                        SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;
                    }
                    Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;   /* Sigmoidal Outputs */
                    /*              Output[p][k] = SumO[p][k];      Linear Outputs */
                    Error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]) ;   /* SSE */
                    DeltaO[k] = DeltaO[k]+(Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;   /* Sigmoidal Outputs, SSE */

                }
          
                /* 'back-propagate' errors to hidden layer*/
                for( j = 1 ; j <= NumHidden ; j++ ) {   
                    SumDOW[j] = 0.0 ;
                    for( k = 1 ; k <= NumOutput ; k++ ) {
                        SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
                    }
                    DeltaH[j] = DeltaH[j]+SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
                }
           
                /* update weights WeightIH */
                for( j = 1 ; j <= NumHidden ; j++ ) { 
                    DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j] ;
                    WeightIH[0][j] += DeltaWeightIH[0][j] ;
                    for( i = 1 ; i <= NumInput ; i++ ) { 
                        DeltaWeightIH[i][j] = eta * Input[p][i] * DeltaH[j] + alpha * DeltaWeightIH[i][j];
                        WeightIH[i][j] += DeltaWeightIH[i][j] ;
                    }
                }
            
            
            
                /* update weights WeightHO */
                for( k = 1 ; k <= NumOutput ; k ++ ) 
                {    
                    DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k] ;
                    WeightHO[0][k] += DeltaWeightHO[0][k] ;
                    for( j = 1 ; j <= NumHidden ; j++ ) 
                    {
                        DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k] ;
                        WeightHO[j][k] += DeltaWeightHO[j][k] ;
                    }

                }
            
            
            }
            //if( epoch%100 == 0 ) fprintf(stdout, "\nEpoch %d :  ", epoch, Error) ;
             epoch++;  
          
    }
    
    fprintf(stdout, "\n\nNETWORK DATA - EPOCH %d - Error %.8f\n\n", epoch, Error) ;    /*print network outputs */
    for( i = 1 ; i <= NumInput ; i++ ) 
    {
        fprintf(stdout, "Input%-4d\t", i) ;
    }
    for( k = 1 ; k <= NumOutput ; k++ ) 
    {
        fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for( p = 1 ; p <= 10 ; p++ ) 
    {        
        printf("\n") ;
        for( i = 1 ; i <= NumInput ; i++ ) 
        {
            fprintf(stdout, "%f\t", Input[p][i]) ;
        }
        for( k = 1 ; k <= NumOutput ; k++ ) 
        {
            fprintf(stdout, "%f\t%f\t", Target[p][k], Output[p][k]) ;
        }
    }
    fprintf(stdout, "\n\nGoodbye!\n\n") ;

}
            


void reinitialise()  // pour reinitialiser le tableau les groupes apres avoir fait le reducer
{
    int i=0;

    for (i = 0; i < nb_groupes; ++i)
    {
        tab_groupe[i].key = 0;
    }


    nb_groupes = 0;
}

        


int main(int argc,char **argv)
{
    
//    afficherListeGroupe();

    int nombre_elements ;
    int  i ,j, rc, t,r,cle; //
    void *status;


    tab_groupe = malloc(sizeof(groupe)*(NUMHID+NUMOUT));
    nb_groupes = 0;


   
    nombre_elements = nombrelignes(argv[1]);// prendre en argument le nom du fichier et retourne le nombre d'element
    NumPattern = nombre_elements-1;
    
    ranpat = malloc(nombre_elements * sizeof(int));
    
    
    Input = (double **)malloc(nombre_elements * sizeof(double *)); // allocation d'un matrice       // on reserve l'espace pour les lignes
    if (Input == NULL)
    {
        fprintf(stderr, "Out of memory");
        exit(0); // si l'allocation echoue on exit
    }
    for (r = 0; r < nombre_elements; r++)
    {// par lignes on alloue les colones
        Input[r] = (double *)malloc((NUMIN+1) * sizeof(double));
        if (Input[r] == NULL)
        {
            fprintf(stderr, "Out of memory");
            exit(0);
        }
    }
    
     Target = (double **)malloc(nombre_elements * sizeof(double *)); // allocation d'un matrice       
    if (Target == NULL)
    {
        fprintf(stderr, "Out of memory");
        exit(0);
    }
    for (r = 0; r < nombre_elements; r++)
    {
        Target[r] = (double *)malloc((NUMOUT+1) * sizeof(double));
        if (Target[r] == NULL)
        {
            fprintf(stderr, "Out of memory");
            exit(0);
        }
    }
    Output = (double **)malloc(nombre_elements * sizeof(double *)); // allocation d'un matrice       
    if (Output == NULL)
    {
        fprintf(stderr, "Out of memory");
        exit(0);
    }
    for (r = 0; r < nombre_elements; r++)
    {
        Output[r] = (double *)malloc((NUMOUT+1) * sizeof(double));
        if (Output[r] == NULL)
        {
            fprintf(stderr, "Out of memory");
            exit(0);
        }
    }
    SumO = (double **)malloc(nombre_elements * sizeof(double *)); // allocation d'un matrice       
    if (SumO == NULL)
    {
        fprintf(stderr, "Out of memory");
        exit(0);
    }
    for (r = 0; r < nombre_elements; r++)
    {
        SumO[r] = (double *)malloc((NUMOUT+1) * sizeof(double));
        if (SumO[r] == NULL)
        {
            fprintf(stderr, "Out of memory");
            exit(0);
        }
    }
    
    
    Hidden = (double **)malloc(nombre_elements * sizeof(double *)); // allocation d'un matrice       
    if (Hidden == NULL)
    {
        fprintf(stderr, "Out of memory");
        exit(0);
    }
    for (r = 0; r < nombre_elements; r++)
    {
        Hidden[r] = (double *)malloc((NUMHID+1) * sizeof(double));
        if (Hidden[r] == NULL)
        {
            fprintf(stderr, "Out of memory");
            exit(0);
        }
    }
    SumH = (double **)malloc(nombre_elements * sizeof(double *)); // allocation d'un matrice       
    if (SumH == NULL)
    {
        fprintf(stderr, "Out of memory");
        exit(0);
    }
    for (r = 0; r < nombre_elements; r++)
    {
        SumH[r] = (double *)malloc((NUMHID+1) * sizeof(double));
        if (SumH[r] == NULL)
        {
            fprintf(stderr, "Out of memory");
            exit(0);
        }
    }
    
    
    tab =malloc(sizeof(paire)*(NUMHID+NUMOUT));
    map_tab=(paire **)malloc( NUM_THREADS* sizeof(paire *));
    
    if (map_tab == NULL)
    {
        fprintf(stderr, "Out of memory");
        exit(0);
    }
    for (r = 0; r <NUM_THREADS; r++)
    {
        map_tab[r] = malloc(sizeof(paire));
        if (map_tab[r] == NULL)
        {
            fprintf(stderr, "Out of memory");
            exit(0);
        }
    }
 

   


	Lecturefichier(argv[1]);//lire le fichier et charger les Input et les Target

    printf("Debut de la vesion sequentielle : \n");
    printf("-----------------------------------");
    
    top1();
     sequential();
    top2();
    unsigned long temps_seq = cpu_time();
	printf("\ntemps sequentiel = %ld.%03ldms\n", temps_seq/1000, temps_seq%1000);
	
    printf("\n");
    printf("\n");

    printf("Debut de la vesion para          : \n");
    printf("-----------------------------------");

    /* initialize WeightIH and DeltaWeightIH */
    for( j = 1 ; j <= NumHidden ; j++ ) 
    {   
        for( i = 0 ; i <= NumInput ; i++ ) { 
            DeltaWeightIH[i][j] = 0.0 ;
            WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
    
    
    
    /* initialize WeightHO and DeltaWeightHO */
    for( k = 1 ; k <= NumOutput ; k ++ ) 
    {    
        for( j = 0 ; j <= NumHidden ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;              
            WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }

    pthread_t thread[NUM_THREADS]; // tableau de thread
	pthread_attr_t attr; 
    pthread_mutex_init(&mutex_sum, NULL);// initialisation du mutex
    // Initialize and set thread detached attribute
    pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);


    Error=1.0;
    epoch =0;
    
    top1();
    while(epoch<EPOCHS )
    {

         

            /* iterate weight updates */
            for( p = 1 ; p <= NumPattern ; p++ ) 
            {   
                /* randomize order of individuals */
                ranpat[p] = p ;
            }
            for( p = 1 ; p <= NumPattern ; p++) 
            {
                np = p + rando() * ( NumPattern + 1 - p ) ;
                op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
            }
        

        
        for(t=0; t<NUM_THREADS; t++) 
        {
		
    		rc = pthread_create(&thread[t], &attr, map, (void *)t);
            // rc permet de savoir si la creation du thread s'est bien passee si oui rc =0
    		if (rc) 
            {
    			printf("ERROR; return code from pthread_create() is %d\n", rc);
    			exit(-1);
    		}
	   } 
        
        for(t=0; t<NUM_THREADS; t++) 
        { // on join les threads afin que le thread maitre attent l'execution des autres avant de ce terminer.
				rc = pthread_join(thread[t], &status);
				if (rc) {
					printf("ERROR; return code from pthread_join() is %d\n", rc);
					exit(-1);
				}
				
        }
             
       /*if( epoch%100 == 0 )
       {
            
            for(t=0; t<NUM_THREADS; t++) 
            {
                tab = map_tab[t];

                for( j = 1 ; j <= NumHidden ; j++ ){
                  printf("cle = %d , value = %f\n",tab[j].key, tab[j].value);
                }

                for( k = 1 ; k <= NumOutput ; k++ ) 
                {
                    cle = k + NUMIN;

                   printf("cle = %d , value = %f\n",tab[cle].key, tab[cle].value);
                }

           } 
        }*/

        regroupement();
        tab_val_groupe(0);
        reduce();
        reinitialise();        
        
        //if( epoch%100 == 0 ) fprintf(stdout, "\nEpoch %d :   ", epoch) ;
        epoch++;        
        
    }  
    
    fprintf(stdout, "\n\nNETWORK DATA - EPOCH %d  \n\n", epoch) ;    /*print network outputs */
    for( i = 1 ; i <= NumInput ; i++ ) 
    {
        fprintf(stdout, "Input%-4d\t", i) ;
    }
    for( k = 1 ; k <= NumOutput ; k++ ) 
    {
        fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for( p = 20 ; p <= 30 ; p++ ) 
    {        
        printf("\n") ;
        for( i = 1 ; i <= NumInput ; i++ ) 
        {
            fprintf(stdout, "%f\t", Input[p][i]) ;
        }
        for( k = 1 ; k <= NumOutput ; k++ ) 
        {
            fprintf(stdout, "%f\t%f\t", Target[p][k], Output[p][k]) ;
        }
    }
    fprintf(stdout, "\n\nGoodbye!\n\n") ;

    top2();
    unsigned long temps_par = cpu_time();
	printf("\ntime parallele = %ld.%03ldms\n", temps_par/1000, temps_par%1000);
	
    return 1 ;
    
}