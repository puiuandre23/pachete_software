options validvarname=v7;

%let cale_csv=/home/u64511646/WineQT.csv;


proc import datafile="&cale_csv"
    out=wine_raw
    dbms=csv
    replace;
    guessingrows=max;
    getnames=yes;
run;



title "Structura setului de date WineQT";
proc contents data=wine_raw;
run;

title "Primele 10 observatii din setul de date";
proc print data=wine_raw(obs=10);
run;


title "Statistici descriptive si valori lipsa";
proc means data=wine_raw n nmiss mean median min max std;
run;


proc format;
    value calitate_fmt
        low - 5 = "Slab/mediu"
        6 - high = "Bun";

    value bun_fmt
        0 = "Slab/mediu"
        1 = "Bun";
run;



data wine_clean;
    set wine_raw;
 
    array valori_numerice _numeric_;
    do i = 1 to dim(valori_numerice);
        if missing(valori_numerice[i]) then valori_numerice[i] = 0;
    end;

    if quality >= 6 then quality_encoded = 1;
    else quality_encoded = 0;

    quality_label = put(quality, calitate_fmt.);
    quality_binara = put(quality_encoded, bun_fmt.);

    if alcohol < 10 then alcohol_class = "Alcool scazut";
    else if alcohol < 12 then alcohol_class = "Alcool mediu";
    else alcohol_class = "Alcool ridicat";

    drop i;
run;


title "Primele 10 observatii dupa prelucrare";
proc print data=wine_clean(obs=10);
    var Id alcohol volatile_acidity sulphates quality quality_label quality_encoded quality_binara alcohol_class;
run;


data vinuri_bune vinuri_slabe_medii;
    set wine_clean;

    if quality_encoded = 1 then output vinuri_bune;
    else output vinuri_slabe_medii;
run;

title "Numar de observatii pe categorii de calitate";
proc freq data=wine_clean;
    tables quality_label quality_encoded quality_binara / nocum;
run;


title "Indicatori medii pe categorii de calitate";
proc means data=wine_clean mean median min max std maxdec=3;
    class quality_label;
    var alcohol volatile_acidity citric_acid sulphates density pH;
run;


title "Analiza prin PROC SQL - medii pe niveluri de calitate";
proc sql;
    select 
        quality,
        count(*) as Numar_observatii,
        mean(alcohol) as Alcool_mediu format=8.3,
        mean(volatile_acidity) as Aciditate_volatila_medie format=8.3,
        mean(sulphates) as Sulfati_medii format=8.3
    from wine_clean
    group by quality
    order by quality;
quit;

title "Cele mai bune vinuri dupa calitate si alcool";
proc sql outobs=10;
    select 
        Id,
        alcohol,
        volatile_acidity,
        sulphates,
        quality,
        quality_label
    from wine_clean
    order by quality desc, alcohol desc;
quit;


title "Distributia calitatii vinurilor";
proc sgplot data=wine_clean;
    vbar quality / datalabel fillattrs=(color=cx4F81BD);
    xaxis label="Calitatea vinului";
    yaxis label="Numar observatii";
run;

title "Relatia dintre alcool si calitatea vinului";
proc sgplot data=wine_clean;
    scatter x=alcohol y=quality / group=quality_label transparency=0.25;
    reg x=alcohol y=quality / lineattrs=(color=red thickness=2);
    xaxis label="Alcool";
    yaxis label="Calitate";
run;

title "Relatia dintre aciditatea volatila si calitatea vinului";
proc sgplot data=wine_clean;
    scatter x=volatile_acidity y=quality / group=quality_label transparency=0.25;
    reg x=volatile_acidity y=quality / lineattrs=(color=red thickness=2);
    xaxis label="Aciditate volatila";
    yaxis label="Calitate";
run;


title "Regresie logistica pentru clasificarea vinurilor bune";
proc logistic data=wine_clean plots(only)=roc;
    class alcohol_class / param=ref;
    model quality_encoded(event='1') = alcohol volatile_acidity sulphates citric_acid density pH;
    output out=wine_predictii p=probabilitate_vin_bun;
run;

title "Primele 10 predictii generate de modelul logistic";
proc print data=wine_predictii(obs=10);
    var Id alcohol volatile_acidity sulphates quality quality_label quality_encoded probabilitate_vin_bun;
run;


title "Regresie liniara multipla - influenta indicatorilor asupra calitatii";
proc reg data=wine_clean;
    model quality = alcohol volatile_acidity sulphates citric_acid density pH;
run;
quit;


title "Final parte SAS - proiect Pachete Software";
proc print data=wine_clean(obs=5);
run;
