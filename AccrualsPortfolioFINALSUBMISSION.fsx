(*
Data Analytics for Finance final project
Total Accruals Investment Strategy f# script

Student: Tomás Borges, 32075

Note: parts of this script were adopted from the PortfolioOptimization.fsx and
PerformanceEvaluation.fsx, while some personal notes have been added throught the script,
some notes were taken from the original files in order to easen the interpretation of this script.
*)

//Imports
#r "nuget:FSharp.Data"
#r "nuget:NodaTime"
#r "nuget: FSharp.Stats"
#r "nuget: Plotly.NET, 2.0.0-beta9"
#r "nuget: FSharp.Stats, 0.4.1"
#r "nuget:Microsoft.ML,1.5"
#r "nuget:Microsoft.ML.MKL.Components,1.5"
#r "nuget: FSharp.Data"
open Microsoft.ML
open Microsoft.ML.Data
open System
open FSharp.Data
open NodaTime
open Plotly.NET
open FSharp.Stats
//open Common

fsi.AddPrinter<DateTime>(fun dt -> dt.ToString("s"))
fsi.AddPrinter<YearMonth>(fun ym -> $"{ym.Year}-{ym.Month}")
// set working directory to script directory
Environment.CurrentDirectory <- __SOURCE_DIRECTORY__
(**
### We will use the portfolio module
*)
#load "Portfolio-3.fsx"
open Portfolio
(**
### We will use the Common.fsx file
*)
#load "../Common.fsx"
open Common
(**
### We need to reference the data
This script assumes the data .csv files  are in the data-cache folder
*)
let [<Literal>] dataCache = __SOURCE_DIRECTORY__ + "/../data-cache/"
let [<Literal>] idAndReturnsFile = dataCache + "id_and_return_data.csv"
let [<Literal>] mySignalFile = dataCache + "taccruals_at.csv"

IO.File.ReadLines(idAndReturnsFile) |> Seq.truncate 5
IO.File.ReadLines(mySignalFile) |> Seq.truncate 5
(**
Define types for the main ID+Return dataset and your signal dataset.
*)
type IdAndReturnCsv = CsvProvider<Sample=idAndReturnsFile,
                                  // Override some column types that I forgot to make boolean
                                  // for easier use when doing filtering.
                                  // If I didn't do this overriding the columns would have strings
                                  // of "1" or "0", but explicit boolean is nicer.
                                  Schema="obsMain(string)->obsMain=bool,exchMain(string)->exchMain=bool",
                                  ResolutionFolder= __SOURCE_DIRECTORY__>

type Signal = CsvProvider<Sample=mySignalFile,
                          ResolutionFolder= __SOURCE_DIRECTORY__>
(**
Loading data. 
*)
let idAndReturns = IdAndReturnCsv.Load(idAndReturnsFile)
let mySignal = Signal.Load(mySignalFile)

let rand = new Random()

let signalDistribution =
    mySignal.Rows
    |> Seq.choose(fun x -> x.Signal)
    // rand.NextDouble() generates a random floating point number
    // between 0 and 1.
    // Here I'm iterating through the sequence and on each iteration
    // I'm generating a random number that I'll use to sort the sequence.
    // It's like Seq.sortBy(fun x -> x.Symbol) but instead of sorting by
    // whatever x.Symbol gives you I'm sorting by whatever
    //  rand.NextDouble() gives you.
    |> Seq.sortBy(fun _ -> rand.NextDouble()) 
    |> Seq.truncate 1_000 
    |> Seq.toArray

let histogram =
    signalDistribution
    |> Chart.Histogram
//  |> Chart.Show 
//By removing the comment bars above, we plot the histogram to get an idea about the distribution of our signal
(**
indexing the data by security id and month.

- `row.Id` as the identifier. We'll assign it to
the `Other` SecurityId case, because it's a dataset specific one.
- In this dataset, the Eom variable defines the "end of month".
- The returns are for the month ending in EOM.
- The signals are "known" as of EOM. So you can use them on/after EOM. We'll
form portfolios in the month ending EOM; that's the `FormationMonth`.
*)
let msfBySecurityIdAndMonth =
    idAndReturns.Rows
    |> Seq.map(fun row -> 
        let index = Other row.Id, YearMonth(row.Eom.Year,row.Eom.Month)
        index, row)
    |> Map.ofSeq    

let signalBySecurityIdAndMonth =
    mySignal.Rows
    |> Seq.choose(fun row -> 
        // we'll use Seq.choose to drop the security if the security is missing. 
        match row.Signal with
        | None -> None // choose will drop these None observations
        | Some signal ->
            let index = Other row.Id, YearMonth(row.Eom.Year,row.Eom.Month)
            Some (index, - signal) // choose will convert Some(index,signal) into
                                 // (index,-signal) and keep that.
                                 //Total Accruals divided by Total Assets should impact returns negatively hence the
                                 //top portfolio will have the lowest taccruals values as we do -signal
    )
    |> Map.ofSeq    
(**
The `securitiesByFormationMonth` that we'll use to define our investment universe.

*)
let securitiesByFormationMonth =
    idAndReturns.Rows
    |> Seq.groupBy(fun x -> YearMonth(x.Eom.Year, x.Eom.Month))
    |> Seq.map(fun (ym, xs) -> 
        ym, 
        xs 
        |> Seq.map(fun x -> Other x.Id) 
        |> Seq.toArray)
    |> Map.ofSeq

let getInvestmentUniverse formationMonth =
    match Map.tryFind formationMonth securitiesByFormationMonth with
    | Some securities -> 
        { FormationMonth = formationMonth 
          Securities = securities }
    | None -> failwith $"{formationMonth} is not in the date range"      
(**
Getting taccruals signal.
*)
let getMySignal (securityId, formationMonth) =
    match Map.tryFind (securityId, formationMonth) signalBySecurityIdAndMonth with
    | None -> None
    | Some signal ->
        Some { SecurityId = securityId; Signal = signal }

let getMySignals (investmentUniverse: InvestmentUniverse) =
    let arrayOfSecuritySignals =
        investmentUniverse.Securities
        |> Array.choose(fun security -> 
            getMySignal (security, investmentUniverse.FormationMonth))    
    
    { FormationMonth = investmentUniverse.FormationMonth 
      Signals = arrayOfSecuritySignals }
(**
getting market capitalization
*)
let getMarketCap (security, formationMonth) =
    match Map.tryFind (security, formationMonth) msfBySecurityIdAndMonth with
    | None -> None
    | Some row -> 
        match row.MarketEquity with
        | None -> None
        | Some me -> Some (security, me)
(**
getting my returns.
*)
let getSecurityReturn (security, formationMonth) =
    // If the security has a missing return, assume that we got 0.0.
    // Note: If we were doing excess returns, we wouldd need 0.0 - rf.
    let missingReturn = 0.0
    match Map.tryFind (security, formationMonth) msfBySecurityIdAndMonth with
    | None -> security, missingReturn
    | Some x ->  
        match x.Ret with 
        | None -> security, missingReturn
        | Some r -> security, r
(**
Adding a few restrictions from the data documentation
section 1.2, "How to use the data":
*)
let isObsMain (security, formationMonth) =
    match Map.tryFind (security, formationMonth) msfBySecurityIdAndMonth with
    | None -> false
    | Some row -> row.ObsMain

let isPrimarySecurity (security, formationMonth) =
    match Map.tryFind (security, formationMonth) msfBySecurityIdAndMonth with
    | None -> false
    | Some row -> row.PrimarySec

let isCommonStock (security, formationMonth) =
    match Map.tryFind (security, formationMonth) msfBySecurityIdAndMonth with
    | None -> false
    | Some row -> row.Common

let isExchMain (security, formationMonth) =
    match Map.tryFind (security, formationMonth) msfBySecurityIdAndMonth with
    | None -> false
    | Some row -> row.ExchMain

let hasMarketEquity (security, formationMonth) =
    match Map.tryFind (security, formationMonth) msfBySecurityIdAndMonth with
    | None -> false
    | Some row -> row.MarketEquity.IsSome

let myFilters securityAndFormationMonth =
    isObsMain securityAndFormationMonth &&
    isPrimarySecurity securityAndFormationMonth &&
    isCommonStock securityAndFormationMonth &&
    isExchMain securityAndFormationMonth &&
    isExchMain securityAndFormationMonth &&
    hasMarketEquity securityAndFormationMonth

let doMyFilters (universe:InvestmentUniverse) =
    let filtered = 
        universe.Securities
        // my filters expect security, formationMonth
        |> Array.map(fun security -> security, universe.FormationMonth)
        // do the filters
        |> Array.filter myFilters
        // now convert back from security, formationMonth -> security
        |> Array.map fst
    { universe with Securities = filtered }
(**
Define sample months
*)
let startSample = 
    idAndReturns.Rows
    |> Seq.map(fun row -> YearMonth(row.Eom.Year,row.Eom.Month))
    |> Seq.min

let endSample = 
    idAndReturns.Rows
    |> Seq.map(fun row -> YearMonth(row.Eom.Year,row.Eom.Month))
    |> Seq.max
    // The end of sample is the last month when we have returns.
    // So the last month when we can form portfolios is one month
    // before that.
    |> fun maxMonth -> maxMonth.PlusMonths(-1) 

let sampleMonths = 
    getSampleMonths (startSample, endSample)
    |> List.toArray
(**
Strategy function
*)
let formStrategy ym =
    ym
    |> getInvestmentUniverse
    |> doMyFilters
    |> getMySignals
    |> assignSignalSort "Mine" 3
    |> Array.map (giveValueWeights getMarketCap)
    |> Array.map (getPortfolioReturn getSecurityReturn)  
(**
taccruals strategy portfolios 
*)
let portfolios =
    sampleMonths
    |> Array.Parallel.collect formStrategy
(**
Common.fsx has some easy to use code to get Fama-French factors.
We're going to use the French data to get monthly risk-free rates.
*)
let ff3 = French.getFF3 Frequency.Monthly
let monthlyRiskFreeRate =
    ff3
    |> Array.map(fun x -> YearMonth(x.Date.Year,x.Date.Month),x.Rf)
    |> Map.ofArray
(**
converting taccruals portfolios into excess returns:
*)
let portfolioExcessReturns =
    portfolios
    |> Array.Parallel.map(fun x -> 
        match Map.tryFind x.YearMonth monthlyRiskFreeRate with 
        | None -> failwith $"Can't find risk-free rate for {x.YearMonth}"
        | Some rf -> { x with Return = x.Return - rf })
(**
plotting the top portfolio
*)
let top =
    portfolioExcessReturns
    |> Array.filter(fun port ->
        port.PortfolioId = Indexed("Mine", 3)) 
    
let cumulateReturn (xs: PortfolioReturn array) =
    let mapper (priorRet:float) (thisObservation:PortfolioReturn) =
        let asOfNow = priorRet*(1.0 + thisObservation.Return)
        { thisObservation with Return = asOfNow}, asOfNow
    // remember to make sure that your sort order is correct.
    let sorted = xs |> Array.sortBy(fun x -> x.YearMonth)
    (1.0, sorted) 
    ||> Array.mapFold mapper 
    |> fst    

let topCumulative = top |> cumulateReturn
(**
Plotly.NET unaware of YearMonth therefore we convert to DateTime before plotting.
*)
let topCumulativeChart =
    topCumulative
    |> Array.map(fun x -> DateTime(x.YearMonth.Year,x.YearMonth.Month,1), x.Return)
    |> Chart.Line 
    |> Chart.withTitle "Growth of 1 Euro - Long-Only Strategy"(* output: 
<null>*)
(**
We could create one normalized to have 10\% annualized volatility
for the entire period. This isn't dynamic rebalancing. We're
just making the whole time-series have 10% vol.
*)
let top10PctVol =
    let topAnnualizedVol = sqrt(12.0) * (top |> Seq.stDevBy(fun x -> x.Return))
    // now lever to have 10% vol.
    // We learned how to do this in volatility timing and it works so long
    // as we have excess returns.
    top 
    |> Array.map(fun x -> { x with Return = (0.1/topAnnualizedVol) * x.Return })
(**
Checking to make sure it's 10\% vol. 
*)
sqrt(12.0) * (top10PctVol |> Seq.stDevBy(fun x -> x.Return)) 
(**
Now plotting 
*)
let topNormalizedPlot =
    top10PctVol
    |> cumulateReturn
    |> Array.map(fun x -> DateTime(x.YearMonth.Year,x.YearMonth.Month,1), x.Return)
    |> Chart.Line 
    |> Chart.withTitle "Growth of 1 Euro"
(**
function to do the vol normalization: 
*)
let normalizeToTenPct (xs:PortfolioReturn array) =
    let annualizedVol = sqrt(12.0) * (xs |> Seq.stDevBy(fun x -> x.Return))
    xs 
    |> Array.map(fun x -> 
        { x with Return = (0.1/annualizedVol) * x.Return })
(**
Check that it does all ports correctly: 
*)
portfolioExcessReturns
|> Array.groupBy(fun port -> port.PortfolioId)
|> Array.map(fun (portId, xs) ->
    let normalized = xs |> normalizeToTenPct  
    portId,
    sqrt(12.0)*(normalized |> Seq.stDevBy(fun x -> x.Return)))
(**
And function to do the plot 
*)
let portfolioReturnPlot (xs:PortfolioReturn array) =
    xs
    |> Array.map(fun x -> DateTime(x.YearMonth.Year,x.YearMonth.Month,1), x.Return)
    |> Chart.Line 
    |> Chart.withTitle "Growth of 1 Euro"
(**
Using the functions: 
*)
let topWithFunctionsPlot =
    top
    |> normalizeToTenPct
    |> cumulateReturn
    |> portfolioReturnPlot
(**
Adding functions to do several portfolios at once. 
First, we add a version of value-weighted market portfolio with same time range and same F# type as our portfolios.
*)
let vwMktRf =
    let portfolioMonths = portfolioExcessReturns |> Array.map(fun x -> x.YearMonth)
    let minYm = portfolioMonths |> Array.min
    let maxYm = portfolioMonths |> Array.max
    
    ff3
    |> Array.map(fun x -> 
        { PortfolioId = Named("Mkt-Rf")
          YearMonth = YearMonth(x.Date.Year,x.Date.Month)
          Return = x.MktRf })
    |> Array.filter(fun x -> 
        x.YearMonth >= minYm &&
        x.YearMonth <= maxYm)
(**
This Chart.Combine method had been shown previously in e.g., the Fred code.
*)
let combinedChart =
    Array.concat [portfolioExcessReturns; vwMktRf]
    |> Array.groupBy(fun x -> x.PortfolioId)
    |> Array.map(fun (portId, xs) ->
        xs
        |> normalizeToTenPct
        |> cumulateReturn
        |> portfolioReturnPlot
        |> Chart.withTraceName (portId.ToString()))
    |> Chart.Combine
(**
You might also want to save your results to a csv file.
*)
type PortfolioReturnCsv = CsvProvider<"portfolioName(string),index(int option),yearMonth(date),ret(float)">

let makePortfolioReturnCsvRow (row:PortfolioReturn) =
    let name, index =
        match row.PortfolioId with
        | Indexed(name, index) -> name, Some index
        | Named(name) -> name, None
    PortfolioReturnCsv
        .Row(portfolioName=name,
             index = index,
             yearMonth=DateTime(row.YearMonth.Year,row.YearMonth.Month,1),
             ret=row.Return)

portfolioExcessReturns
|> Array.map makePortfolioReturnCsvRow
|> fun rows -> 
    let csv = new PortfolioReturnCsv(rows)
    csv.Save("myExcessReturnPortfolios.csv")
(*
To construct the strategy – We will sort stocks monthly into terciles based on your signal 
(Bottom 1/3, Middle 1/3, Top 1/3). 
Stocks in the “bottom” should have worst expected performance since they have the highest level of Total Accruals divided by Total Assets
stocks in the “top” should be those with the best expected performance. 
We'll now Form value-weighted portfolios for each of your terciles. The returns for these terciles are “excess returns”, 
meaning returns in excess of the risk-free rate (ri − rf ). 
The top tercile portfolio is your long-only strategy portfolio. 
We'll also form a long-short strategy portfolio that is long the top portfolio and short the bottom portfolio
*)
top

top |> Array.map (fun x -> x.Return) |> Chart.Histogram |> Chart.withTitle "Distribution of daily returns top tercile" |> Chart.Show 
//aparent skewness to the left given the presence of extreme left tail events

topCumulativeChart
|> Chart.Show

(**
Let's plot the Middle 1/3 portfolio
*)
let middle =
    portfolioExcessReturns
    |> Array.filter(fun port ->
        port.PortfolioId = Indexed("Mine", 2))

let middleCumulative = middle |> cumulateReturn
(**
Plotly.NET doesn't know about YearMonth, so I will convert to DateTime before plotting.
*)
let middleCumulativeChart =
    middleCumulative
    |> Array.map(fun x -> DateTime(x.YearMonth.Year,x.YearMonth.Month,1), x.Return)
    |> Chart.Line 
    |> Chart.withTitle "Growth of 1 Euro - middle tercile"

middleCumulativeChart
|> Chart.Show

(**
Let's plot the Bottom 1/3 portfolio
*)
let bottom =
    portfolioExcessReturns
    |> Array.filter(fun port ->
        port.PortfolioId = Indexed("Mine", 1))

let bottomCumulative = bottom |> cumulateReturn
(**
Plotly.NET doesn't know about YearMonth, so I will convert to DateTime before plotting.
*)
let bottomCumulativeChart =
    bottomCumulative
    |> Array.map(fun x -> DateTime(x.YearMonth.Year,x.YearMonth.Month,1), x.Return)
    |> Chart.Line 
    |> Chart.withTitle "Growth of 1 Euro - bottom tercile"

bottomCumulativeChart
|> Chart.Show

//PLOTING LONGSHORT PORTFOLIO CUMULATIVE RETURNS (note: here we build according the the top portfolio logic, but for the portfolio optimization part we do so through the portfolio return csv)
let shortPortMap =
    bottom
    |> Array.map (fun x -> x.YearMonth, x)
    |> Map.ofArray

let longShortPortReturns =
    top
    |> Array.choose (fun longOb ->
        let matchingHedgeReturn = Map.tryFind longOb.YearMonth shortPortMap
        match matchingHedgeReturn with
        | None -> None
        | Some hedgeOb ->
            let longShort = longOb.Return - hedgeOb.Return
            Some { PortfolioId = Indexed("Long Short", 4); YearMonth = longOb.YearMonth; Return = longShort}) //index = 4 is arbitrary

let longshortCumulative = longShortPortReturns |> cumulateReturn
(**
Plotly.NET doesn't know about YearMonth, so I will convert to DateTime before plotting.
*)
let longshortCumulativeChart =
    longshortCumulative
    |> Array.map(fun x -> DateTime(x.YearMonth.Year,x.YearMonth.Month,1), x.Return)
    |> Chart.Line 
    |> Chart.withTitle "Growth of 1 Euro - Long&Short Strategy"

longshortCumulativeChart
|> Chart.Show


//NOTE: Top corresponds to the long only portfolio
top
//NOTE: excess returns of the value-weighted stock market portfolio (from the Ken French data, MktRf) is given by:
vwMktRf
let vwMktRfcum = vwMktRf |> cumulateReturn
let vwMktRfChart =
    vwMktRfcum
    |> Array.map(fun x -> DateTime(x.YearMonth.Year,x.YearMonth.Month,1), x.Return)
    |> Chart.Line 
    |> Chart.withTitle "Growth of 1 Euro"

//PLOTING LONG-ONLY, LONG-SHORT, and Value-weight mkt portfolio TOGETHER
Chart.Combine [longshortCumulativeChart |> Chart.withTraceName(Name="Long-Short Strategy");
   topCumulativeChart |> Chart.withTraceName(Name="Long-Only Strategy");
   vwMktRfChart |> Chart.withTraceName(Name="value-weighted stock market portfolio")] 
|> Chart.withTitle "Growth of 1 Euro - Long-only vs Long-Short vs vwMktRf"
|> Chart.Show

//NORMALISING OUR PORTFOLIOS
//long only is the normalized top portfolio:
//topNormalizedPlot |> Chart.Show

let longshort10PctVol = normalizeToTenPct longShortPortReturns

sqrt(12.0) * (longshort10PctVol |> Seq.stDevBy(fun x -> x.Return)) //checks out

let longshortNormalizedPlot =
    longshort10PctVol
    |> cumulateReturn
    |> Array.map(fun x -> DateTime(x.YearMonth.Year,x.YearMonth.Month,1), x.Return)
    |> Chart.Line 
    |> Chart.withTitle "Growth of 1 Euro"

//I dont run the function "portfolioReturnPlot" because i want the accumulated returns

let vwMktRf10PctVol = normalizeToTenPct vwMktRf
let vwMktRfNormalizedPlot =
    vwMktRf10PctVol
    |> cumulateReturn
    |> Array.map(fun x -> DateTime(x.YearMonth.Year,x.YearMonth.Month,1), x.Return)
    |> Chart.Line 
    |> Chart.withTitle "Growth of 1 Euro"

//PLOTING NORMALISED PORTFOLIOS TOGETHER
Chart.Combine [longshortNormalizedPlot |> Chart.withTraceName(Name="Long-Short Normalised");
   topNormalizedPlot |> Chart.withTraceName(Name="Long-Only Normalised");
   vwMktRfNormalizedPlot |> Chart.withTraceName(Name="vwMktRf Normalised")] 
|> Chart.withTitle "Growth of 1 Euro - Portfolios at annualized volatility of 10%"
|> Chart.Show


//1st half, 2nd half, full period
let half xs = //gives months of 2nd half 
    xs
    |> Array.map (fun x -> x.YearMonth)
    |> Array.splitInto 2
    |> Array.last

half longShortPortReturns
half top
half vwMktRf
printfn $"We know for the 3 portfolios that 2010-8 is the 1st month of the second half"
//defining a variable with constituent months of second half
let monthofsplit = half longShortPortReturns

let firstHalf xs = xs |> Array.filter (fun x -> x.YearMonth < monthofsplit.[0]) 
let secondHalf xs = xs |> Array.filter (fun x -> x.YearMonth >= monthofsplit.[0])
//1st half
let firstHalf_longShortPortReturns = firstHalf longShortPortReturns
let firstHalf_top = firstHalf top
let firstHalf_vwMktRf = firstHalf vwMktRf
//2nd half
let secondHalf_longShortPortReturns = secondHalf longShortPortReturns
let secondHalf_top = secondHalf top
let secondHalf_vwMktRf = secondHalf vwMktRf
 
(*For each period report: 
∗ What is their average annualized return? 
∗ What are their annualized Sharpe ratios? *)
let retannualizer (xs : PortfolioReturn []) =
    xs
    |> Array.map (fun xs -> xs.Return) 
    |> Array.average
    |> fun x -> x * 12.
//1st half
let annret_firstHalf_longShortPortReturns = retannualizer firstHalf_longShortPortReturns
let annret_firstHalf_top = retannualizer firstHalf_top
let annret_firstHalf_vwMktRf = retannualizer firstHalf_vwMktRf
//2nd half
let annret_secondHalf_longShortPortReturns = retannualizer secondHalf_longShortPortReturns
let annret_secondHalf_top = retannualizer secondHalf_top
let annret_secondHalf_vwMktRf = retannualizer secondHalf_vwMktRf
//full period
let annret_longShortPortReturns = retannualizer longShortPortReturns
let annret_top = retannualizer top
let annret_vwMktRf = retannualizer vwMktRf

//RUNNING VARIABLES ABOVE GIVES ANNUALISED EXCESS RETURNS

//Sharpe Ratios
let annSRgiver (positions : PortfolioReturn []) =
    let annrets = 
        retannualizer positions
    let annvol =
        positions
        |> Array.map (fun xs -> xs.Return)
        |> Seq.stDev |> (fun x -> x* sqrt(12.))
    annrets / annvol
//1st half
let annSR_firstHalf_longShortPortReturns = annSRgiver firstHalf_longShortPortReturns
let annSR_firstHalf_top = annSRgiver firstHalf_top
let annSR_firstHalf_vwMktRf = annSRgiver firstHalf_vwMktRf
//2nd half
let annSR_secondHalf_longShortPortReturns = annSRgiver secondHalf_longShortPortReturns
let annSR_secondHalf_top = annSRgiver secondHalf_top
let annSR_secondHalf_vwMktRf = annSRgiver secondHalf_vwMktRf
//full period
let annSR_longShortPortReturns = annSRgiver longShortPortReturns
let annSR_top = annSRgiver top
let annSR_vwMktRf = annSRgiver vwMktRf

//RUNNING VARIABLES ABOVE GIVES ANNUALISED SHARPE RATIOS


//Portfolio evaluation code below Modified for taccruals

(**
For regression, it is helpful to have the portfolio
return data merged into our factor model data.
*)
type RegData =
    // The ML.NET OLS trainer requires 32bit "single" floats
    { Date : DateTime
      Portfolio : single
      MktRf : single 
      Hml : single 
      Smb : single }

// ff3 indexed by month
// We're not doing date arithmetic, so I'll just
// use DateTime on the 1st of the month to represent a month
let ff3ByMonth = 
    ff3
    |> Array.map(fun x -> DateTime(x.Date.Year, x.Date.Month,1), x)
    |> Map

let longShortRegData =
    longShortPortReturns 
    |> Array.map(fun port ->
        let monthToFind = DateTime(port.YearMonth.Year,port.YearMonth.Month,1)
        match Map.tryFind monthToFind ff3ByMonth with
        | None -> failwith "probably you messed up your days of months"
        | Some ff3 -> 
            { Date = monthToFind
              Portfolio = single port.Return // single converts to 32bit
              MktRf = single ff3.MktRf 
              Hml = single ff3.Hml 
              Smb = single ff3.Smb })
(**
We need to define a ML.Net. Once instantiated by the user, it provides a way to create components for data preparation, feature enginering, training, prediction, model evaluation.
*)
let ctx = new MLContext()
(**
We use the context to transform the data into ML.NET's format.
*)
let longShortMlData = ctx.Data.LoadFromEnumerable<RegData>(longShortRegData)
(**
Now we are going to define our machine learning trainer: OLS!
*)
let trainer = ctx.Regression.Trainers.Ols()
(**
Now we define the models that we want to estimate: capm and ff3. Same logic as ML pipeline
- `Label` - intrepreted as ML target; `Features` - variables to predict target
*)
let capmModel = 
    EstimatorChain()
        .Append(ctx.Transforms.CopyColumns("Label","Portfolio"))
        .Append(ctx.Transforms.Concatenate("Features",[|"MktRf"|])) 
        .Append(trainer)   

let ff3Model =
    EstimatorChain()
        .Append(ctx.Transforms.CopyColumns("Label","Portfolio"))
        .Append(ctx.Transforms.Concatenate("Features",[|"MktRf";"Hml";"Smb"|]))
        .Append(trainer)   
(**
Now we can estimate our models.
*)
let capmEstimate = longShortMlData |> capmModel.Fit
let ff3Estimate = longShortMlData |> ff3Model.Fit
(**
CAPM results.
*)
capmEstimate.LastTransformer.Model
(**
Fama-French 3-Factor model results
*)
ff3Estimate.LastTransformer.Model
(**
probably the CAPM $R^2$ is lower than the Fama-French $R^2$. therefore we can explain more of the portfolio's returns with the Fama-French model. Or in trader terms,
you can hedge the portfolio better with the multi-factor model.
We also want predicted values so that we can get regression residuals for calculating
the information ratio. ML.NET calls the predicted value the [score] -> see microsoft notation examples about ML.net
*)
[<CLIMutable>]
type Prediction = { Label : single; Score : single}

let makePredictions (estimate:TransformerChain<_>) data =
    ctx.Data.CreateEnumerable<Prediction>(estimate.Transform(data),reuseRowObject=false)
    |> Seq.toArray

let residuals (xs: Prediction array) = xs |> Array.map(fun x -> x.Label - x.Score)

let capmPredictions = makePredictions capmEstimate longShortMlData
let ff3Predictions = makePredictions ff3Estimate longShortMlData

let capmResiduals = residuals capmPredictions
let ff3Residuals = residuals ff3Predictions
(**
We'll later create functions to these following provided by the professor:
*)
let capmAlpha = (single 12.0) * capmEstimate.LastTransformer.Model.Bias 
let capmStDevResiduals = sqrt(single 12) * (Seq.stDev capmResiduals)
let capmInformationRatio = capmAlpha / capmStDevResiduals

let ff3Alpha = (single 12.0) * ff3Estimate.LastTransformer.Model.Bias 
let ff3StDevResiduals = sqrt(single 12) * (Seq.stDev ff3Residuals)
let ff3InformationRatio = ff3Alpha / ff3StDevResiduals

// Function version

let informationRatio monthlyAlpha (monthlyResiduals: single array) =
    let annualAlpha = single 12.0 * monthlyAlpha
    let annualStDev = sqrt(single 12.0) * (Seq.stDev monthlyResiduals)
    annualAlpha / annualStDev 

informationRatio capmEstimate.LastTransformer.Model.Bias capmResiduals
informationRatio ff3Estimate.LastTransformer.Model.Bias ff3Residuals

//∗ What are their CAPM and Fama-French 3-factor alphas and t-statistics for these alphas? ∗ What are their information ratios?
//Example above was only for short long so we build a function here that does all the heavy work for us based on the portfolio's PortfolioReturn [] we built in the 1st part
let estimator3000 (xs : PortfolioReturn []) =
    let portfolioRegData xs =
        xs 
        |> Array.map(fun port ->
            let monthToFind = DateTime(port.YearMonth.Year,port.YearMonth.Month,1)
            match Map.tryFind monthToFind ff3ByMonth with
            | None -> failwith "probably you messed up your days of months"
            | Some ff3 -> 
                { Date = monthToFind
                  Portfolio = single port.Return // single converts to 32bit
                  MktRf = single ff3.MktRf 
                  Hml = single ff3.Hml 
                  Smb = single ff3.Smb })
    let portfolioMlData = ctx.Data.LoadFromEnumerable<RegData>(portfolioRegData xs)
    let capmEstimate = portfolioMlData |> capmModel.Fit
    let ff3Estimate = portfolioMlData |> ff3Model.Fit
    let capmodelestimates = capmEstimate.LastTransformer.Model
    let threefactormodelestimates = ff3Estimate.LastTransformer.Model

    let makePredictions (estimate:TransformerChain<_>) data =
        ctx.Data.CreateEnumerable<Prediction>(estimate.Transform(data),reuseRowObject=false)
        |> Seq.toArray

    let residuals (xs: Prediction array) = xs |> Array.map(fun x -> x.Label - x.Score)

    let capmPredictions = makePredictions capmEstimate longShortMlData
    let ff3Predictions = makePredictions ff3Estimate longShortMlData

    let capmResiduals = residuals capmPredictions
    let ff3Residuals = residuals ff3Predictions

    let capmAlpha = (single 12.0) * capmEstimate.LastTransformer.Model.Bias 
    let capmStDevResiduals = sqrt(single 12) * (Seq.stDev capmResiduals)
    let capmInformationRatio = capmAlpha / capmStDevResiduals 

    let ff3Alpha = (single 12.0) * ff3Estimate.LastTransformer.Model.Bias 
    let ff3StDevResiduals = sqrt(single 12) * (Seq.stDev ff3Residuals)
    let ff3InformationRatio = ff3Alpha / ff3StDevResiduals

    let informationRatio monthlyAlpha (monthlyResiduals: single array) =
        let annualAlpha = single 12.0 * monthlyAlpha
        let annualStDev = sqrt(single 12.0) * (Seq.stDev monthlyResiduals)
        annualAlpha / annualStDev 

    let capminformationRatio = informationRatio capmEstimate.LastTransformer.Model.Bias capmResiduals
    let ff3informationRatio = informationRatio ff3Estimate.LastTransformer.Model.Bias ff3Residuals
    [capmodelestimates, threefactormodelestimates, [capmAlpha, ff3Alpha], [capminformationRatio, ff3informationRatio]]
//Important note on estimator3000: 
//first couple elements return capm and ff3 ols parameters, 
//third element returns list with capm and ff3 Alphas
//& last element two returned values are the information ratios for capm and ff3
//THEORETICAL NOTE: t-stats (given by TVALUES) are constant for different timeframes

//Alpha, Alpha t-stat and Information Ratio ANALYSIS BY PORTFOLIO
//alphas returned in the 3rd element
//t-stats found in capm and ff3 model estimates TValues (first Tvalue returned relates to alpha)
//information ratios returned by last elements

//1st half
estimator3000 firstHalf_longShortPortReturns
//CAPM: alpha= -0.0222804267 ,t-stat= -1.084830787,Information Ratio = -0.3560779691
//FF3: alpha= -0.01460780483,t-stat= -0.6921153596,Information Ratio = -0.2419652939
estimator3000 firstHalf_top
//CAPM: alpha= -0.01299808547 ,t-stat= -0.8556722053,Information Ratio = -0.06692744046
//FF3: alpha= -0.008664863184,t-stat= -0.5965949092,Information Ratio = -0.0449594073
estimator3000 firstHalf_vwMktRf
//CAPM: alpha= -4.542107142e-12 ,t-stat= -infinity,Information Ratio = -2.71509551e-11
//FF3: alpha= -1.250502046e-11,t-stat= -infinity,Information Ratio = -7.475016439e-11
//2nd half
estimator3000 secondHalf_longShortPortReturns
//CAPM: alpha= 0.03358587995 ,t-stat= 1.850069495,Information Ratio = 0.5311065912
//FF3: alpha= 0.01273600012,t-stat= 0.7588625883,Information Ratio = 0.2054619044
estimator3000 secondHalf_top
//CAPM: alpha= 0.009126708843 ,t-stat= 0.8017509147,Information Ratio = 0.05320332199
//FF3: alpha= -6.957065489e-05,t-stat= -0.00642389502,Information Ratio = -0.0004009208933
estimator3000 secondHalf_vwMktRf
//CAPM: alpha= 8.523382e-11 ,t-stat= infinity,Information Ratio = 5.094947242e-10
//FF3: alpha= 1.12210255e-10,t-stat= infinity,Information Ratio = 6.707494005e-10
//full period
estimator3000 longShortPortReturns
//CAPM: alpha= 5.399686779e-05 ,t-stat= 0.003930706433,Information Ratio = 0.0008678712766
//FF3: alpha= 0.002200157847,t-stat= 0.1668107121,Information Ratio = 0.03702093661
estimator3000 top
//CAPM: alpha= -0.008868858218 ,t-stat= -0.9089139926,Information Ratio = -0.04800844938
//FF3: alpha= -0.008326169103,t-stat= -0.9289845929,Information Ratio = -0.0452192463
estimator3000 vwMktRf
//CAPM: alpha= 3.287646544e-11 ,t-stat= infinity,Information Ratio = 1.965227564e-10
//FF3: alpha= 2.957039821e-11,t-stat= infinity,Information Ratio = 1.767603702e-10


(*  'Strategy as part of a diversified portfolio'
• Form mean-variance optimal diversified portfolios for your strategy
1st long-only portfolio + other assets (VTI & BND)  2nd long-short portfolio + other assets
For long portfolios, make sure that you are using excess returns. 
– Use the full time-period to estimate average returns, variances, and covariances of all assets. 
For means and variances, use all available data for that asset. 
For covariances/correlations, use the mutually overlapping time period 
(e.g., if one asset has data starting in 2000 and another starting in 2005, 
estimate the covariance or correlation from 2005 onward)
*)

//The code below's foundation is the Portfolio optimization script, we use it as the core to what will be our Strategy as part of a diversified portfolio

//Some imports already present in top of the script, nevertheless it's interesting to see what we need for what follows
#r "nuget: FSharp.Stats, 0.4.1" 
#r "nuget: FSharp.Data"

#load "common.fsx"

open System
open FSharp.Data
open Common

open FSharp.Stats


Environment.CurrentDirectory <- __SOURCE_DIRECTORY__
(**
# Portfolio Optimization
*)
type StockData =
    { Symbol : string 
      Date : DateTime
      Return : float }
(**
We get the Fama-French 3-Factor asset pricing model data.
*)
//let ff3 = French.getFF3 Frequency.Monthly <- variable already called

// Transform to a StockData record type.
let ff3StockData =
    [| 
       ff3 |> Array.map(fun x -> {Symbol="HML";Date=x.Date;Return=x.Hml})
       ff3 |> Array.map(fun x -> {Symbol="MktRf";Date=x.Date;Return=x.MktRf})
       ff3 |> Array.map(fun x -> {Symbol="Smb";Date=x.Date;Return=x.Smb})
    |] |> Array.concat
(**
Let's get our factor data.
*)
let myFactorPorts = CsvProvider<"myExcessReturnPortfolios.csv",
                                ResolutionFolder = __SOURCE_DIRECTORY__>.GetSample()

let long = 
    myFactorPorts.Rows 
    |> Seq.filter(fun row -> row.PortfolioName = "Mine" && row.Index = Some 3)
    |> Seq.map(fun x -> { Symbol = "Long"; Date = x.YearMonth; Return = x.Ret })
    |> Seq.toArray

//This time i'll display stock returns of the Long-short porfolio in a different way
//To fit the portfolio optimization code, i aim to import the long short portfolio as 'StockData []'

let short = 
    myFactorPorts.Rows 
    |> Seq.filter(fun row -> row.PortfolioName = "Mine" && row.Index = Some 1)
    |> Seq.map(fun x -> { Symbol = "Long"; Date = x.YearMonth; Return = x.Ret })
    |> Seq.toArray

let longshort = 
    let short = short |> Seq.map(fun row -> row.Date, row) |> Map
    long
    |> Seq.map (fun topObs ->
        match Map.tryFind topObs.Date short with
        | None -> failwith "probably your date variables are not aligned"
        | Some shortObs -> { Symbol = "Long Short"; Date = topObs.Date; Return = topObs.Return - shortObs.Return })
    |> Seq.toArray  //exactly same returns as longShortPortReturns but using a different Type

(**
Some good standard investments.
*)
let vti = //Vanguard Total Stock Market ETF 
    "VTI"
    |> Tiingo.request
    |> Tiingo.startOn (DateTime(2000,1,1))
    |> Tiingo.getReturns

let bnd = //Vanguard Total Bond Market ETF
    "BND"
    |> Tiingo.request
    |> Tiingo.startOn (DateTime(2000,1,1))
    |> Tiingo.getReturns
(**
These are daily returns. So let's convert them to monthly returns.
*)

(**
So let's combine the `VTI` and `BND` data, group by symbol and month,
and then convert to monthly returns.
*)
let standardInvestments =
    Array.concat [vti; bnd]
    |> Array.groupBy(fun x -> x.Symbol, x.Date.Year, x.Date.Month)
    |> Array.map(fun ((sym, year, month), xs) -> 
        let sortedRets = 
            xs
            |> Array.sortBy(fun x -> x.Date)
            |> Array.map(fun x -> x.Return)
        let monthlyGrossRet =
            (1.0, sortedRets)
            ||> Array.fold (fun acc x -> acc * (1.0 + x))
        { Symbol = sym
          Date = DateTime(year, month, 1)
          Return = monthlyGrossRet - 1.0 })
(**
And let's convert to excess returns
*)
let rf = ff3 |> Seq.map(fun x -> x.Date, x.Rf) |> Map

let standardInvestmentsExcess =
    let maxff3Date = ff3 |> Array.map(fun x -> x.Date) |> Array.max
    standardInvestments
    |> Array.filter(fun x -> x.Date <= maxff3Date)
    |> Array.map(fun x -> 
        match Map.tryFind x.Date rf with 
        | None -> failwith $"why isn't there a rf for {x.Date}"
        | Some rf -> { x with Return = x.Return - rf })
(**
If we did it right, the `VTI` return should be pretty similar to the `MktRF`
return from Ken French's website.
*)
standardInvestmentsExcess
|> Array.filter(fun x -> x.Symbol = "VTI" && x.Date.Year = 2021)
|> Array.map(fun x -> x.Date.Month, round 4 x.Return)
|> Array.take 3(* output: 
val it : (int * float) [] = [|(1, -0.0033); (2, 0.0314); (3, 0.0365)|]*)
ff3 
|> Array.filter(fun x -> x.Date.Year = 2021)
|> Array.map(fun x -> x.Date.Month, round 4 x.MktRf)(* output: 
val it : (int * float) [] = [|(1, -0.0003); (2, 0.0278); (3, 0.0309)|]*)

(**
Let's put our stocks in a map keyed by symbol
*)
let stockData (portfolio : StockData []) =
    Array.concat [| standardInvestmentsExcess; portfolio |]
    |> Array.groupBy(fun x -> x.Symbol)
    |> Map

let symbols stockData = 
    stockData 
    |> Map.toArray // convert to array of (symbol, observations for that symbol) 
    |> Array.map fst // take just the symbol
    |> Array.sort // sort them
(**
Let's create a function that calculates covariances
for two securities.
*)
let getCov x y stockData =
    let innerJoin xId yId =
        let xRet = Map.find xId stockData
        let yRet = Map.find yId stockData |> Array.map(fun x -> x.Date, x) |> Map
        xRet
        |> Array.choose(fun x ->
            match Map.tryFind x.Date yRet with
            | None -> None
            | Some y -> Some (x.Return, y.Return))
    let x, y = innerJoin x y |> Array.unzip
    Seq.cov x y

let covariances symbols stockData = //Now needs to take the entire symbols function to take different portfolios
    symbols
    |> Array.map(fun x ->
        symbols
        |> Array.map(fun y -> getCov x y stockData))
    |> matrix

let means stockData = //Now needs to take the stockData as a variable
    stockData
    |> Map.toArray
    |> Array.map(fun (sym, xs) ->
        sym,
        xs |> Array.averageBy(fun x -> x.Return))
    |> Array.sortBy fst
    |> Array.map snd
    |> vector


let information_portfolio (portfolio : StockData []) = //fits long and longshort
    //calls functions above to get a collection of info on portfolio
    let stockData = stockData portfolio
    let symbols = symbols stockData
    let covariances = covariances symbols stockData
    let means = means stockData

    let w' = Algebra.LinearAlgebra.SolveLinearSystem covariances means
    let w = w' |> Vector.map(fun x -> x /  Vector.sum w')
    let portVariance = w.Transpose * covariances * w
    let portStDev = sqrt(portVariance)
    let portMean = w.Transpose * means
    let info_anon_record =
        {|Weights = w; Mean = w.Transpose * means; Variance = w.Transpose * covariances * w; Covariance = covariances|}
    info_anon_record


let Longinfo = information_portfolio long
let LongShortinfo = information_portfolio longshort

(**
## Comparing mean-variance efficient to 60/40.

Now let's form the mean-variance efficient portfolios based on the above optimal weights and compare them to a 60/40 portfolio over our sample. A 60% stock and 40%
bond portfolio is a common investment portfolio. Our weights are sorted by `symbols`. Let's put them into a Map collection for easier referencing.
*)

let weights portfolio =
    let stockData = stockData portfolio
    let information_portfolio = information_portfolio portfolio
    let w = information_portfolio.Weights
    Seq.zip (symbols stockData) w
    |> Map.ofSeq

let weightsLong = weights long
let weightsLonhShort = weights longshort

(**
Next, we'd like to get the symbol data grouped by date.
*)
let full_returner_by_date portfolio =
    let stockDataByDate =
        let stockData = stockData portfolio
        stockData
        |> Map.toArray // Convert to array of (symbol, StockData)
        |> Array.map snd // grab only the stockData from (symbol, StockData)
        |> Array.collect id // combine all different StockData symbols into one array.
        |> Array.groupBy(fun x -> x.Date) // group all symbols on the same date together.
        |> Array.sortBy fst // sort by the grouping variable, which here is Date.
    let firstMonth =
        stockDataByDate 
        |> Array.head // first date group
        |> snd // convert (date, StockData array) -> StockData array
    let lastMonth =
        stockDataByDate 
        |> Array.last // last date group
        |> snd // convert (date, StockData array) -> StockData array
    let symbols = symbols (stockData portfolio) //required for both allAssetsStart & allAssetsEnd
    let allAssetsStart =
        stockDataByDate
        // find the first array element where there are as many stocks as you have symbols
        |> Array.find(fun (month, stocks) -> stocks.Length = symbols.Length)
        |> fst // convert (month, stocks) to month
    let allAssetsEnd =
        stockDataByDate
        // find the last array element where there are as many stocks as you have symbols
        |> Array.findBack(fun (month, stocks) -> stocks.Length = symbols.Length)
        |> fst // convert (month, stocks) to month
    stockDataByDate
    |> Array.filter(fun (date, stocks) -> 
        date >= allAssetsStart &&
        date <= allAssetsEnd)



(**
Now let's make my mve and 60/40 ports
Now in a function that takes weights and monthData as input
*)
let portfolioMonthReturn weights monthData =
    weights
    |> Map.toArray
    |> Array.map(fun (symbol, weight) ->
        let symbolData = 
            // we're going to be more safe and use tryFind here so
            // that our function is more reusable
            match monthData |> Array.tryFind(fun x -> x.Symbol = symbol) with
            | None -> failwith $"You tried to find {symbol} in the data but it was not there"
            | Some data -> data
        symbolData.Return*weight)
    |> Array.sum    
    
(**
Here's a thought. We just made a function that takes weights and a month as input. That means that it should work if we give it different weights.
Let's try to give it 60/40 weights.
*)
let weights6040 = Map [("VTI",0.6);("BND",0.4)]
//We can use these weights later on to make the 60/40 portfolio

(**
Now we're ready to make our mve and 60/40 portfolios.
*)
let portMve portfolio =
    full_returner_by_date portfolio
    |> Array.map(fun (date, data) -> 
        { Symbol = "MVE"
          Date = date
          Return = portfolioMonthReturn (weights portfolio) data })
let port6040 portfolio = 
    full_returner_by_date portfolio
    |> Array.map(fun (date, data) -> 
        { Symbol = "60/40"
          Date = date 
          Return = portfolioMonthReturn weights6040 data} )

//We now define our MVE and 60/40 portfolio returns
//long only
let long_returns_mve = portMve long
let long_returns_6040 = port6040 long
//long short
let longshort_returns_mve = portMve longshort
let longshort_returns_6040 = port6040 longshort

(**
cumulative returns.
*)
#r "nuget: Plotly.NET, 2.0.0-beta9"
open Plotly.NET
(**
A function to accumulate returns.
*)
let cumulateReturns xs =
    let mapFolder prevRet x =
        let newReturn = prevRet * (1.0+x.Return)
        { x with Return = newReturn}, newReturn
    
    (1.0, xs) 
    ||> Array.mapFold mapFolder
    |> fst    
(**
Ok, cumulative returns. We'll use a function that works for both the MVE and 60/40
*)
let portCumulative portMveOr6040 = //portMveOr6040 despite complicated name is just to point we can use both cases
    portMveOr6040
    |> cumulateReturns

//to plot the chart of each case we first define a charting function
let chartinator portfolio (title : string) = 
    portCumulative portfolio
    |> Array.map(fun x -> x.Date, x.Return)
    |> Chart.Line
    |> Chart.withTraceName title

//Another function to combine different charts -> Most importantly combines 6040 and MVE but can have later uses
let chartCombined chartA chartB =
    [| chartA; chartB |]
    |> Chart.Combine


let chart_longonly = chartCombined (chartinator long_returns_mve "Long Only MVE") (chartinator long_returns_6040 "60/40")
let chart_longshort = chartCombined (chartinator longshort_returns_mve "Long-Short MVE") (chartinator longshort_returns_6040 "60/40")

chart_longonly |> Chart.withTitle "Cumulative Returns with Long-Only portfolio" //|> Chart.Show
chart_longshort |> Chart.withTitle "Cumulative Returns with Long-Short portfolio" //|> Chart.Show
//We can always just chart all three portfolios together to make an easy to visualise panel
chartCombined chart_longonly (chartinator longshort_returns_mve "Long-Short MVE") |> Chart.withTitle "Cumulative Returns" |> Chart.Show

(**
Those are partly going to differ because they have different volatilities.
It we want to have a sense for which is better per unit of volatility,
then it can make sense to normalize volatilities.

cumulative returns of the normalized vol returns.
*)
let normalize10pctVol xs =
    let vol = xs |> Array.map(fun x -> x.Return) |> Seq.stDev
    let annualizedVol = vol * sqrt(12.0)
    xs 
    |> Array.map(fun x -> { x with Return = x.Return * (0.1/annualizedVol)})

//The following functions can be applied to cases of both MVE and 60/40
let portfolioCumulativeNormlizedVol portMveOr6040 = 
    portMveOr6040
    |> normalize10pctVol
    |> cumulateReturns

let chartportfolioNormlizedVol portMveOr6040 (title : string) = //already does the heavy work and returns cum_rets of normalised portfolios
    portfolioCumulativeNormlizedVol portMveOr6040
    |> Array.map(fun x -> x.Date, x.Return)
    |> Chart.Line
    |> Chart.withTraceName title

//To combine the charts of the (un)leveraged to 10% vol MVE & 60/40s we use chartCombined

let chart_longonly_normalised = chartCombined (chartportfolioNormlizedVol long_returns_mve "Long Only MVE Normalised") (chartportfolioNormlizedVol long_returns_6040 "60/40 Normalised")
let chart_longshort_normalised = chartCombined (chartportfolioNormlizedVol longshort_returns_mve "Long-Short MVE Normalised") (chartportfolioNormlizedVol longshort_returns_6040 "60/40 Normalised")

chart_longonly_normalised |> Chart.withTitle "Cumulative Returns with constant leverage (set to 10% ann.volatility) - with Long-Only portfolio" //|> Chart.Show
chart_longshort_normalised |> Chart.withTitle "Cumulative Returns with constant leverage (set to 10% ann.volatility) - with Long-Short portfolio" //|> Chart.Show
//We can always just chart all three portfolios together to make an easy to visualise panel
chartCombined chart_longonly_normalised (chartportfolioNormlizedVol longshort_returns_mve "Long-Short MVE normalised") |> Chart.withTitle "Cumulative Returns with constant leverage (set to 10% ann.volatility)" |> Chart.Show

(* 
Now that we built our strategy, we will end by analysing its performance in terms of 
average annualized return and annualized Sharpe ratio 
*)

//Easiest way is to define a function that returns a series of statistics per porfolio

//For the average returns and variances we use similar methods as when we calculated the SRs for our longonly and longshort portfolios
let retannualiser_full_strat port = 
    port
    |> Array.map (fun xs -> xs.Return) 
    |> Array.average
    |> fun x -> x * 12.

let annstd_full_strat port =
    port
    |> Array.map (fun xs -> xs.Return)
    |> Seq.stDev |> (fun x -> x* sqrt (12.))

let SR_full_strat port =
    (retannualiser_full_strat port) / (annstd_full_strat port)

//Annualised returns
//MVE
let annualised_long_returns_mve = retannualiser_full_strat long_returns_mve
let annualised_longshort_returns_mve = retannualiser_full_strat longshort_returns_mve
//60/40
let annualised_long_returns_6040 = retannualiser_full_strat long_returns_6040
let annualised_longshort_returns_6040 = retannualiser_full_strat longshort_returns_6040

//Annualised Sharpe ratios
//MVE
let annSR_long_returns_mve = SR_full_strat long_returns_mve
let annSR_longshort_returns_mve = SR_full_strat longshort_returns_mve
//60/40
let annSR_long_returns_6040 = SR_full_strat long_returns_6040
let annSR_longshort_returns_6040 = SR_full_strat longshort_returns_6040


(* values for the table which reports performance measures for the portfolios over the full period:
val annualised_long_returns_mve : float = 0.0401995426
val annualised_longshort_returns_mve : float = 0.04487675248
val annualised_long_returns_6040 : float = 0.07406227466
val annualised_longshort_returns_6040 : float = 0.07406227466
val annSR_long_returns_mve : float = 1.024461014
val annSR_longshort_returns_mve : float = 1.068550487
val annSR_long_returns_6040 : float = 0.7416210048
val annSR_longshort_returns_6040 : float = 0.7416210048
*)
