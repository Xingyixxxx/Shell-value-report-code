* =====================================================================
* 借壳概率预测：数据预处理（CSMAR 原始数据 → data/raw/original.csv）
* 只需修改下面一行 ROOT 为本机项目根目录
* 原始数据放在 $ROOT/data/csmar/<子目录>/<文件>.dta，中间文件写到 $ROOT/data/interim/
* =====================================================================
global ROOT ""
global RAW  "$ROOT/data/csmar"
global X    "$ROOT/data/interim"
capture mkdir "$X"

*********被解释变量 Y：借壳事件*********
* listed_company.dta：手工整理的被借壳公司名单，date 为借壳年份
* 每个事件展开为事件年及其前 5 年共 6 行，事件年 Y=1，其余为 0
use "$RAW/listed_company.dta", clear
gen year = year(date)
expand 6
bysort Stkcd (year): gen time = year - _n + 1
keep Stkcd principle time year
gen Y = 0
replace Y = 1 if year == time
order Stkcd time Y year principle
sort Stkcd time
save "$X/listed_company.dta", replace

*********TOP1：第一大股东持股比例*********
use "$RAW/Top1/BDT_ExcessiveDebt.dta", clear
gen date = date(Enddate, "YMD")
format date %td
gen time = year(date)
keep if ST == 0
keep if IsNewOrSuspend == 0
keep if ISBSE == 1
rename Symbol Stkcd
keep Stkcd LargestHolderRate time
rename LargestHolderRate TOP1
save "$X/Top1.dta", replace

*********Soe：产权性质*********
use "$RAW/Soe/BDT_ManaGovAbil.dta", clear
gen date = date(Enddate, "YMD")
format date %td
gen time = year(date)
rename Symbol Stkcd
rename PropertyRightsNature Soe
keep Stkcd time Soe
save "$X/Soe.dta", replace

*********Size：年末流通市值对数*********
use "$RAW/Size_Ret/TRD_Year.dta", clear
gen date = date(Trdynt, "Y")
gen time = year(date)
gen Size = log(Ysmvosd)
keep Stkcd Size time Markettype
save "$X/Size.dta", replace

*********Sales：营业收入增长率*********
use "$RAW/Sales/FI_T8.dta", clear
keep if Typrep == "A"
gen date = date(Accper, "YMD")
gen time = year(date)
bysort Stkcd time(date): keep if _n == _N
rename F081602C Sales
keep Stkcd time Sales
save "$X/Sales.dta", replace

*********Op、Roe：营业利润率、净资产收益率*********
use "$RAW/Op_Roe/FI_T5.dta", clear
keep if Typrep == "A"
gen date = date(Accper, "YMD")
gen time = year(date)
bysort Stkcd time(date): keep if _n == _N
rename F050504C Roe
rename F051401C Op
keep Stkcd time Op Roe
save "$X/Op_Roe.dta", replace

*********IPO：当年 IPO 上会家数与通过率*********
use "$RAW/IPO/IPO_ISSUEAUDIT.dta", clear
rename MeetingYear time
bysort time: gen count = _N
gen succeed = .
replace succeed = 1 if AuditConclusion == "1"
bysort time: gen scount = sum(succeed)
bysort time: keep if _n == _N
gen IPO = scount/count
rename count IPOnum
keep time IPOnum IPO
save "$X/IPO.dta", replace

*********Ind、Lev：行业代码、资产负债率*********
use "$RAW/Ind_Lev/BDT_FinIndex.dta", clear
rename Symbol Stkcd
gen date = date(EndDate, "YMD")
gen time = year(date)
keep if StateType == "A"
rename AssetLiabilityRatio Lev
rename IndustryCode1 Ind
keep Stkcd time Ind Lev
save "$X/Ind_Lev.dta", replace

*********Cash：现金持有率*********
use "$RAW/Cash/FI_T3.dta", clear
gen date = date(Accper, "YMD")
gen time = year(date)
keep if Typrep == "A"
bysort Stkcd time(date): keep if _n == _N
rename F030201A Cash
keep Stkcd Cash time
save "$X/Cash.dta", replace

*********Insider、reverrate：高管持股比例、流通股比例*********
use "$RAW/Insider/HLD_Capstru.dta", clear
gen Insider = Nshrsms/Nshrttl
gen reverrate = Nshrn/Nshrttl
gen date = date(Reptdt, "YMD")
gen time = year(date)
bysort Stkcd time(date): keep if _n == _N
keep Stkcd time Insider reverrate
save "$X/Insider.dta", replace

*********ST：是否 ST/*ST*********
use "$RAW/上市公司状态/STK_LISTEDCOINFOANL.dta", clear
gen date = date(EndDate, "YMD")
gen time = yofd(date)
rename Symbol Stkcd
gen ST = 0
replace ST = 1 if LISTINGSTATE == "*ST" | LISTINGSTATE == "ST"
drop if LISTINGSTATE == "终止上市"
keep Stkcd time ST
save "$X/ST.dta", replace

*********return：年度超额收益率 = 个股年收益率 - 市场年收益率*********
use "$RAW/Year_Stkcd_return/TRD_Year.dta", clear
gen time = yearly(Trdynt, "Y")
rename Yretwd year_return
keep Stkcd time year_return Yarkettype
save "$X/year_stkcd_return.dta", replace
use "$RAW/Year_Market_return/TRD_Yearm.dta", clear
gen time = yearly(Trdynt, "Y")
rename Yretwdos year_market_return
rename Markettype Yarkettype
keep time year_market_return Yarkettype
save "$X/year_market_return.dta", replace
use "$X/year_stkcd_return.dta", clear
merge m:1 Yarkettype time using "$X/year_market_return.dta"
gen return = year_return - year_market_return
keep Stkcd time return
save "$X/year_special_return.dta", replace

*********合并面板*********
use "$X/Size.dta", clear
foreach f in Cash Ind_Lev Insider {
    merge 1:1 Stkcd time using "$X/`f'.dta"
    drop if _merge == 2
    drop _merge
}
merge m:1 time using "$X/IPO.dta"
drop if _merge == 2
drop _merge
foreach f in Op_Roe Sales Soe Top1 listed_company ST year_special_return {
    merge 1:1 Stkcd time using "$X/`f'.dta"
    drop if _merge == 2
    drop _merge
}

* 只保留上证 A 股（Markettype=1）和深证 A 股（Markettype=4）主板
keep if Markettype == 1 | Markettype == 4
drop if time < 2000 | time > 2023

replace Y = 0 if Y == .
* 行业缺失时用同一公司前后年份的行业补齐
bysort Stkcd (time): replace Ind = Ind[_n-1] if missing(Ind) & _n > 1
bysort Stkcd (time): replace Ind = Ind[_n+1] if missing(Ind) & _n < _N
* 保留借壳事件发生之后的观测（Y=0），用作对比（报告 3.1、9.6）
drop year principle Markettype
* 剔除金融业
drop if strpos(Ind, "J") > 0

* O 门类（居民服务业）公司统一使用其 2013 年的行业代码
gen O_ind = 1 if strpos(Ind, "O") > 0
preserve
keep if time == 2013
keep Stkcd Ind
duplicates drop
rename Ind Ind_2013
tempfile ind2013
save `ind2013'
restore
merge m:1 Stkcd using `ind2013'
replace Ind = Ind_2013 if O_ind == 1
drop O_ind Ind_2013 _merge

*********构造滞后变量*********
encode Stkcd, gen(stkcd)
xtset stkcd time
* _3：t-3 至 t-1 年的均值（rangestat 对窗口内非缺失年份取平均）
foreach v in Size return Op Sales reverrate Insider Lev Cash Roe TOP1 {
    rangestat (mean) `v'_3 = `v', interval(time -3 -1) by(Stkcd)
}
* Soe、ST 缺失时用前两年的取值补齐（前两年任一年为 1 则记为 1）
rangestat (mean) Soe_2 = Soe, interval(time -2 -1) by(Stkcd)
replace Soe_2 = 1 if Soe_2 > 0 & Soe_2 != .
replace Soe = Soe_2 if Soe == .
rangestat (mean) ST_2 = ST, interval(time -2 -1) by(Stkcd)
replace ST_2 = 1 if ST_2 > 0 & ST_2 != .
replace ST = ST_2 if ST == .
* _1：t-1 年取值
foreach v in Size return Op Sales Insider reverrate Lev Cash IPO IPOnum Soe ST Roe TOP1 {
    gen `v'_1 = L.`v'
}
keep if time > 2002
keep Stkcd time Y Ind Size_1 return_1 Op_1 Sales_1 Insider_1 reverrate_1 Lev_1 Cash_1 IPO_1 IPOnum_1 Soe_1 ST_1 Roe_1 TOP1_1 ///
     Size_3 return_3 Op_3 Sales_3 reverrate_3 Insider_3 Lev_3 Cash_3 Roe_3 TOP1_3
order Stkcd time Y Ind
* 连续变量在 1% 和 99% 分位缩尾
winsor2 Size_1 Op_1 Sales_1 return_1 Insider_1 reverrate_1 Lev_1 Cash_1 IPO_1 IPOnum_1 Roe_1 TOP1_1 ///
        Size_3 return_3 Op_3 Sales_3 reverrate_3 Insider_3 Lev_3 Cash_3 Roe_3 TOP1_3, cuts(1 99) replace
export delimited using "$ROOT/data/raw/original.csv", replace

*********月度 IPO 统计（描述性统计作图用）*********
use "$RAW/IPO/IPO_ISSUEAUDIT.dta", clear
gen date = date(DeclareDate, "YMD")
format date %td
gen month = mofd(date)
format month %tm
bysort month: gen count = _N
gen succeed = .
replace succeed = 1 if AuditConclusion == "1"
bysort month: gen scount = sum(succeed)
bysort month: keep if _n == _N
keep month count scount
gen rate = scount/count
export excel using "$ROOT/data/raw/IPO.xlsx", firstrow(variables) replace
