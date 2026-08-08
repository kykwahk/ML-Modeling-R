
####################################################
## R을 이용한 머신러닝: 모델링, Tidymodels, Caret ##
## (곽기영, 도서출판 청람)                        ## 
####################################################

#######################
## 부록 C. Tidyverse ##
#######################

library(tidyverse)

################
## C.1 tibble ##
################

library(tibble)
tibble(id=c("A001", "A002", "A003"), 
       name=c("Mouse", "Keyboard", "USB"), 
       price=c(30000, 90000, 50000))

library(modeldata)
str(credit_data)
as_tibble(credit_data)

tibble(price=c(10, 22, 35),
       quantity=c(30, 50, 15),
       sales=price*quantity)

tbl <- tibble(id=1:3,
              data=list(tibble(x=1, y=2),
                        tibble(y=c(3, 4), z=c(5, 6)),
                        tibble(z=c(7, 8, 9), x=c(10, 11, 12))))
tbl
tbl$data
tbl$data[[2]]

###############
## C.2 dplyr ##
###############

library(tibble)
airtbl <- as_tibble(airquality)
airtbl

library(dplyr)
filter(airtbl, Month == 7)

filter(airtbl, Month == 7, Temp > 90)
filter(airtbl, Month == 7 & Temp > 90)
filter(airtbl, Ozone > 80 | Temp > 90)

slice(airtbl, 6:10)

slice_head(airtbl, n=3)
slice_tail(airtbl, n=3)
slice_head(airtbl)
slice_tail(airtbl)

slice_max(airtbl, order_by=Ozone, n=3)
slice_min(airtbl, order_by=Temp, n=3)
slice_max(airtbl, order_by=Ozone)
slice_min(airtbl, order_by=Temp)
slice_min(airtbl, order_by=Temp, n=3, with_ties=FALSE)

select(airtbl, 5, 6, 4)
select(airtbl, Month, Day, Temp)
select(airtbl, Temp:Day)
select(airtbl, -(Ozone:Wind))

select(airtbl, Month, Day, everything())

?select

arrange(airtbl, Temp, Month, Day)
arrange(airtbl, desc(Temp), Month, Day)

mutate(airtbl, Temp.C=(Temp-32)/1.8, Diff=Temp.C - mean(Temp.C))

summarise(airtbl, mean(Temp))
summarise(airtbl, 
          Min=min(Temp, na.rm=TRUE),
          Median=median(Temp, na.rm=TRUE),
          Mean=mean(Temp, na.rm=TRUE),
          SD=sd(Temp, na.rm=TRUE),
          Max=max(Temp, na.rm=TRUE),
          N=n(),
          Distinct.Month=n_distinct(Month),
          First.Month=first(Month),
          Last.Month=last(Month))	

air.group <- group_by(airtbl, Month)
class(air.group)
air.group

summarise(air.group,
          Number.of.Days=n(),
          Mean.Temp=mean(Temp, na.rm=TRUE),
          SD.Temp=sd(Temp, na.rm=TRUE))

air.ungroup <- ungroup(air.group)
class(air.ungroup)
summarise(air.ungroup, Mean.Temp=mean(Temp, na.rm=TRUE))

count(airtbl, Month, sort=TRUE)

tbl1 <- tibble(x=1:6, y=month.name[1:6])
tbl2 <- tibble(x=7:12, y=month.name[7:12])
tbl3 <- bind_rows(tbl1, tbl2)
tbl3

tbl4 <- tibble(z=month.abb)
bind_cols(tbl3, tbl4)

band_members
band_instruments

inner_join(x=band_members, y=band_instruments, by="name")

left_join(band_members, band_instruments, by="name")
right_join(band_members, band_instruments, by="name")
full_join(band_members, band_instruments, by="name")

band_instruments2
full_join(band_members, band_instruments2, by=c("name"="artist"))

semi_join(band_members, band_instruments, by="name")

anti_join(band_members, band_instruments, by="name")

1:10 %>% mean()
1:10 |> mean()

a1 <- select(airquality, Ozone, Temp, Month)
a2 <- group_by(a1, Month)
a3 <- summarise(a2, 
                Mean.Ozone=mean(Ozone, na.rm=TRUE), 
                Mean.Temp=mean(Temp, na.rm=TRUE))
a4 <- filter(a3, Mean.Ozone > 30 | Mean.Temp > 70)
a5 <- arrange(a4, desc(Mean.Temp))
a6 <- left_join(a5, tibble(Month=1:12, Month.Name=month.name), by="Month")
a6

air <- airtbl %>% 
  select(Ozone, Temp, Month) %>% 
  group_by(Month) %>% 
  summarise(Mean.Ozone=mean(Ozone, na.rm=TRUE), 
            Mean.Temp=mean(Temp, na.rm=TRUE)) %>% 
  filter(Mean.Ozone > 30 | Mean.Temp > 70) %>% 
  arrange(desc(Mean.Temp)) %>% 
  left_join(tibble(Month=1:12, Month.Name=month.name), by="Month")
air

###############
## C.3 purrr ##
###############

exams <- list(s1=c(78, 89, 91, 85, 95, 98),
              s2=c(85, 86, 97, 99, 90),
              s3=c(98, 96, 89, 90, 93, 85, 92),
              s4=c(98, 96, 91, 88, 93, 99))
exams
library(purrr)
map(.x=exams, .f=mean)

map_dbl(exams, mean)

map_dbl(exams, mean, trim=0.3)

exams |>  
  map_dbl(mean, trim=0.3)

map(exams, function(.) .*1.1) 

map(exams, ~.*1.1) 
map(exams, ~.x*1.1) 

fruits <- c("Apple", "Banana", "Strawberry")
fruits |> 
  map_chr(~paste(.x, "Juice", sep="-"))

# [그림 C-1]
windows(width=7.0, height=5.5)
par(mfrow=c(2, 2))
walk(exams, hist)

# [그림 C-2]
windows(width=7.0, height=5.5)
par(mfrow=c(2, 2))
iwalk(exams, ~hist(.x, main=.y, xlab="Score"))

a <- list(1, 2, 3)
b <- list(10, 20, 30)
map2(.x=a, .y=b, .f=function(first, second) second - first)

map2(.x=a, .y=b, ~.y-.x)

map2_dbl(a, b, ~.y-.x)

set.seed(123)
map2(b, a, rnorm, n=5)

a <- list(1, 2, 3)
b <- list(10, 20, 30)
c <- list(100, 200, 300)
pmap(.l=list(a, b, c), .f=function(first, second, third) second - first + third)

pmap_dbl(list(a, b, c), ~..2-..1+..3)

args <- expand.grid(n=c(50, 100),
                    mean=c(1, 10),
                    sd=c(1, 5))
args
set.seed(123)
pmap(args, rnorm)

# [그림 C-3]
windows(width=7.0, height=5.5)
par(mfrow=c(2, 4))
set.seed(123)
pmap(args, rnorm) |>  
  iwalk(~hist(.x, main=paste("Sample", .y), , xlab="X"))

###############
## C.4 tidyr ##
###############

library(tidyr)
head(airquality)
aq.long <- pivot_longer(data=airquality, cols=Ozone:Temp,
                        names_to="Factor", values_to="Measurement")
head(aq.long)
tail(aq.long)

pivot_longer(data=airquality, cols=Ozone:Temp, 
             names_to="Factor", values_to="Measurement")
pivot_longer(data=airquality, cols=1:4, 
             names_to="Factor", values_to="Measurement")
pivot_longer(data=airquality, cols=c(-Month, -Day), 
             names_to="Factor", values_to="Measurement")

aq.wide <- pivot_wider(data=aq.long, names_from=Factor, values_from=Measurement)
head(aq.wide)
tail(aq.wide)

#################
## C.5 ggplot2 ##
#################

library(modeldata)
str(penguins)
levels(penguins$species)

library(ggplot2)

# [그림 C-6]
windows(width=7.0, height=5.5)
ggplot(data=penguins, mapping=aes(x=bill_length_mm, y=bill_depth_mm)) +
  geom_point() 

# [그림 C-7]
windows(width=7.0, height=5.5)
ggplot(penguins, aes(x=bill_length_mm, y=bill_depth_mm)) +
  geom_point(pch=21, color="black", fill="salmon", alpha=0.7, size=2) +
  geom_density_2d(color="purple") +
  geom_smooth(color="cornflowerblue", linewidth=1.5) +
  labs(title="Penguin's Bill",
       x="Bill Length (mm)", y="Bill Depth (mm)") +
  theme_bw()

?theme

# [그림 C-8]
p1 <- ggplot(penguins, aes(x=bill_length_mm, y=bill_depth_mm, shape=species)) +
  geom_point(size=2) +
  theme_bw() +
  theme(legend.title=element_blank(),
        legend.position="bottom")
p2 <- ggplot(penguins, aes(x=bill_length_mm, y=bill_depth_mm, color=species)) +
  geom_point(size=2) +
  theme_bw() +
  theme(legend.title=element_blank(),
        legend.position="bottom")
windows(width=10.0, height=5.5)
library(patchwork)
p1 + p2

# [그림 C-9]
windows(width=7.0, height=5.5)
ggplot(penguins, aes(x=bill_length_mm, y=bill_depth_mm, shape=species, color=species)) +
  geom_point(size=2) +
  scale_color_brewer(palette="Set2") +
  theme_bw() +
  theme(legend.title=element_blank(),
        legend.position="bottom")

library(RColorBrewer)
?RColorBrewer
?scale_color_viridis_d

# [그림 C-10]
windows(width=7.0, height=5.5)
ggplot(penguins, aes(x=bill_length_mm, y=bill_depth_mm)) +
  facet_wrap(vars(species)) +
  geom_point(pch=21, color="black", fill="cornflowerblue", size=2) +
  theme_bw() +
  theme(strip.text=element_text(size=10, face="bold", color="darkblue"),
        strip.background=element_rect(fill="azure"))
