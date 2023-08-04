using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace CourseApp1.data
{
    [MetadataType(typeof(TraineeMetadata))]
    public partial class Trainee
    {
    }
    public class TraineeMetadata
    {
        public int ID { get; set; }
        [Display(Name ="Student Name")]
        public string Name { get; set; }
        public string Email { get; set; }
        public string Password { get; set; }
        public Nullable<bool> Is_active { get; set; }
    }

}